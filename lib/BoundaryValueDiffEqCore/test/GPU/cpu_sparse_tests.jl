include("device_sparse_tests.jl")

# A backend without a sparse adapter must still take the dense fallback.
# CPU views inherit sparse support from their parent storage.
struct DenseOnlyArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(x::DenseOnlyArray) = size(x.data)
Base.getindex(x::DenseOnlyArray, i::Int) = x.data[i]
Base.similar(x::DenseOnlyArray, dims::Dims) = similar(x.data, dims)

@testset "CPU sparse storage" begin
    test_device_sparse_storage(identity)
    template = view(zeros(4), 1:2)
    @test __device_sparse_supported(template)
    @test __device_sparse_matrix(template, spzeros(2, 2)).matrix isa SparseMatrixCSC
    unsupported = DenseOnlyArray(zeros(2))
    @test !__device_sparse_supported(unsupported)
    @test_throws ArgumentError __device_sparse_matrix(unsupported, spzeros(2, 2))

    pattern = sparse([1, 2], [1, 2], [1.0, 2.0])
    storage = __device_sparse_matrix(zeros(2), pattern)
    @test storage.matrix.colptr !== pattern.colptr
    @test rowvals(storage.matrix) !== rowvals(pattern)
    @test nonzeros(storage.matrix) !== nonzeros(pattern)
end

using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector
using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore, BVPJacobianAlgorithm
using KernelAbstractions: CPU
using SciMLBase: StandardBVProblem, StandardSecondOrderBVProblem, TwoPointBVProblem,
    TwoPointSecondOrderBVProblem

# An analytic nonlinear residual exercises the shared pipeline independently of
# any solver's collocation formula or automatic differentiation implementation.
struct AnalyticCollocationCache{A, P, J}
    alg::A
    problem_type::P
    jacobian_cache::J
    residual::Base.RefValue{Vector{Float64}}
    y::Matrix{Float64}
    device_cache::Dict{DataType, Any}
    resid_size::Tuple{Tuple{Int}, Tuple{}}
    reuse::Vector{Bool}
end

BoundaryValueDiffEqCore.__bvp_device_unknowns(cache::AnalyticCollocationCache) = cache.y
function BoundaryValueDiffEqCore.__device_residual!(
        r, u, cache::AnalyticCollocationCache, boundary = true, reuse = false
    )
    push!(cache.reuse, reuse)
    boundary && (r[1] = u[1]^2 + u[4])
    r[2] = u[1] * u[2]
    r[3] = sin(u[2]) + u[3]
    r[4] = u[3]^2 - u[4]
    return r
end

@testset "Shared collocation Jacobians" begin
    core = BoundaryValueDiffEqCore
    u = [0.3, -0.7, 1.2, 0.4]
    expected = [2u[1] 0 0 1; u[2] u[1] 0 0; 0 cos(u[2]) 1 0; 0 0 2u[3] -1]
    pattern = sparse(expected)
    modes = (
        AutoForwardDiff(; chunksize = 2), AutoFiniteDiff(),
        AutoFiniteDiff(; fdjtype = Val(:central)),
    )
    for problem_type in (
                StandardBVProblem(), StandardSecondOrderBVProblem(),
                TwoPointBVProblem{true}(), TwoPointSecondOrderBVProblem{true}(),
            ),
            mode in modes, sparse_storage in (false, true)
        jac_alg = BVPJacobianAlgorithm(mode)
        y = reshape(copy(u), 1, :)
        storage = if sparse_storage
            core.__prepare_device_jacobian(y, problem_type, ((1,), (0,))) do
                group = core.__bvp_device_sparse_group(pattern, 1:4, mode, true)
                (; pattern, groups = (group,), boundary_fallback = false)
            end
        else
            (; matrix = zeros(4, 4), plan = nothing)
        end
        cache = AnalyticCollocationCache(
            (; platform = CPU(), jac_alg), problem_type, Ref(storage.plan),
            Ref(zeros(4)), y, Dict{DataType, Any}(), ((1,), ()), Bool[]
        )
        core.__device_jacobian!(storage.matrix, u, cache)
        @test Matrix(storage.matrix) ≈ expected rtol = 1.0e-6 atol = 1.0e-7
        workspaces = collect(values(cache.device_cache))
        core.__device_jacobian!(storage.matrix, u, cache)
        @test all(a === b for (a, b) in zip(workspaces, values(cache.device_cache)))
        mode isa AutoForwardDiff && !sparse_storage && @test any(cache.reuse)
        products = core.__device_jacobian_products(cache)
        if sparse_storage
            direction, out = [2.0, -1.0, 0.5, 3.0], zeros(4)
            products.jvp(out, direction, u, nothing)
            @test out ≈ expected * direction rtol = 1.0e-6
            products.vjp(out, direction, u, nothing)
            @test out ≈ expected' * direction rtol = 1.0e-6
        else
            @test products == (; jvp = nothing, vjp = nothing)
        end
    end

    # Boundary AD and central differences for collocation use separate row slices.
    jac_alg = BVPJacobianAlgorithm(;
        bc_diffmode = modes[1], nonbc_diffmode = modes[3]
    )
    cache = AnalyticCollocationCache(
        (; platform = CPU(), jac_alg), StandardBVProblem(), Ref(nothing),
        Ref(zeros(4)), reshape(copy(u), 1, :), Dict{DataType, Any}(), ((1,), ()), Bool[]
    )
    J = fill(NaN, 4, 4)
    core.__device_jacobian!(J, u, cache)
    @test J ≈ expected rtol = 1.0e-8 atol = 1.0e-9

    # Known patterns must preserve structural zeros and bypass boundary tracing.
    known = sparse([1, 1], [1, 4], [0.0, 1.0], 1, 4)
    mode = AutoSparse(modes[1]; sparsity_detector = KnownJacobianSparsityDetector(known))
    traced, fallback = core.__device_boundary_pattern(mode, Float64, 1, 4) do
        error("Known sparsity must not construct the boundary callback")
    end
    @test !fallback
    @test nnz(traced) == 2
    traced, fallback = core.__device_boundary_pattern(modes[1], Float64, 1, 4) do
        error("Unsupported tracing")
    end
    @test fallback
    @test Matrix(traced) == trues(1, 4)

    storage = core.__prepare_device_jacobian(
        DenseOnlyArray(reshape(copy(u), 1, :)), StandardBVProblem(), ((1,), ())
    ) do
        error("Dense backends must not construct sparse metadata")
    end
    @test storage.plan === nothing
    @test size(storage.matrix) == (4, 4)
end

@testset "Shared device parameter and shape dispatch" begin
    core = BoundaryValueDiffEqCore
    parameters = (; rate = [2.0], nested = (3.0, [4.0, 5.0]))
    adapted = core.__device_parameter(CPU(), parameters)
    @test adapted == parameters
    @test adapted.rate !== parameters.rate
    @test adapted.nested[2] !== parameters.nested[2]
    parameters.rate[1] = 6
    core.__device_copy_parameter!(adapted, parameters)
    @test adapted == parameters
    @test_throws DimensionMismatch core.__device_copy_parameter!(adapted.rate, [1.0, 2.0])

    data = collect(1.0:6.0)
    shaped = core.__device_reshape(data, (2, 3))
    @test shaped == reshape(data, 2, 3)
    shaped[2, 3] = 7
    @test data[6] == 7
    @test core.__device_reshape(data, (6,)) === data
    @test core.__device_bc_sizes(StandardBVProblem(), nothing, shaped) == ((2, 3), ())
    @test core.__device_bc_sizes(StandardSecondOrderBVProblem(), nothing, shaped) == ((12,), ())
    @test core.__device_initial_backend(view(data, 1:2)) isa CPU
end
