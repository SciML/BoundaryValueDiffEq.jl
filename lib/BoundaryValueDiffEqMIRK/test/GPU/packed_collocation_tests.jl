using BoundaryValueDiffEqMIRK
using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector
using BoundaryValueDiffEqMIRK: ForwardDiff
using SciMLBase: BVPFunction, BVProblem, TwoPointBVProblem, init, isinplace, solve, solve!, successful_retcode
using SparseArrays: SparseMatrixCSC, nnz, nonzeros
using Test

const MIRK = BoundaryValueDiffEqMIRK

function packed_test_f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p.rate[1] * sin(u[1]) + t
    return nothing
end
packed_test_f(u, p, t) = (u[2], -p.rate[1] * sin(u[1]) + t)
function packed_test_bc!(res, u, p, t)
    res[1] = u(0.0)[1]
    res[2] = u(1.0)[1] - 1
    return nothing
end
packed_test_bc(u, p, t) = [u(0.0)[1], u(1.0)[1] - 1]

function packed_test_matrix!(du, u, p, t)
    for j in 1:2
        du[1, j] = u[2, j]
        du[2, j] = -p.rate[1] * sin(u[1, j]) + t
    end
    return nothing
end

function packed_test_residual(cache, device_cache, u, packed; constraint = Val(false))
    trait = MIRK.__cache_trait(cache.alg.jac_alg)
    y = MIRK.recursive_unflatten!(cache.y, u)
    residual_size = constraint isa Val{true} ? length(cache.f_prototype) : cache.M
    residual = [similar(u, residual_size) for _ in eachindex(cache.mesh_dt)]
    if packed
        MIRK.__mirk_device_collocation!(residual, device_cache, cache, y, u, trait, constraint)
    elseif isinplace(cache.prob)
        MIRK.Φ!(residual, cache, y, u, trait, constraint)
    else
        residual = MIRK.Φ(cache, y, u, trait)
    end
    return reduce(vcat, residual)
end

function packed_test_cache(prob, alg; dt = 0.2)
    cache = init(prob, alg; dt, adaptive = false)
    device_cache = MIRK.__mirk_device_cache_impl(
        CPU(), prob, cache.alg, prob.u0, cache.TU, haskey(prob.kwargs, :tune_parameters)
    )
    return cache, device_cache
end

@testset "Packed collocation on the CPU backend" begin
    # Exercise device packing and kernels on the CPU backend.
    @testset "$Alg / inplace=$iip" for Alg in (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6, MIRK6I),
            iip in (true, false)
        prob = BVProblem(
            iip ? packed_test_f! : packed_test_f,
            iip ? packed_test_bc! : packed_test_bc,
            [0.3, 0.7], (0.0, 1.0), (; rate = [1.2])
        )
        cache, device_cache = packed_test_cache(prob, Alg())
        @test cache.device_cache === nothing
        u = copy(cache.y₀_flat)
        u .+= range(0.0, 0.4; length = length(u))
        cpu_residual = packed_test_residual(cache, device_cache, u, false)
        trait = MIRK.__cache_trait(cache.alg.jac_alg)
        cpu_stages = [copy(MIRK._collocation_tmp(k, u, trait)) for k in cache.k_discrete]
        device_residual = packed_test_residual(cache, device_cache, u, true)
        @test device_residual ≈ cpu_residual rtol = 1.0e-13 atol = 1.0e-14
        for (k, cpu_k) in zip(cache.k_discrete, cpu_stages)
            @test MIRK._collocation_tmp(k, u, trait) ≈ cpu_k rtol = 1.0e-13 atol = 1.0e-14
        end

        if Alg === MIRK4
            cpu_loss = x -> packed_test_residual(cache, device_cache, x, false)
            device_loss = x -> packed_test_residual(cache, device_cache, x, true)
            # Distinct chunk widths exercise separate cached Dual element types.
            for chunk in (ForwardDiff.Chunk{1}(), ForwardDiff.Chunk{3}())
                cpu_config = ForwardDiff.JacobianConfig(cpu_loss, u, chunk)
                device_config = ForwardDiff.JacobianConfig(device_loss, u, chunk)
                cpu_jac = ForwardDiff.jacobian(cpu_loss, u, cpu_config)
                device_jac = ForwardDiff.jacobian(device_loss, u, device_config)
                @test device_jac ≈ cpu_jac rtol = 1.0e-12 atol = 1.0e-13
            end
            @test packed_test_residual(cache, device_cache, u, true) ≈ cpu_residual

            # Refresh mutable arrays and a redistributed mesh even when their
            # sizes have not changed between residual evaluations.
            prob.p.rate[1] = 2.0
            cache.mesh[2] = 0.13
            cache.mesh_dt .= diff(cache.mesh)
            expected = packed_test_residual(cache, device_cache, u, false)
            @test expected != cpu_residual
            @test packed_test_residual(cache, device_cache, u, true) ≈ expected

            # Mesh refinement must replace buffers allocated for the previous N.
            refined = init(prob, Alg(); dt = 0.1, adaptive = false)
            refined_u = copy(refined.y₀_flat)
            @test packed_test_residual(refined, device_cache, refined_u, true) ≈
                packed_test_residual(refined, device_cache, refined_u, false)
        end
    end

    @testset "Matrix state and singular term" begin
        singular_matrix = [
            0.0 0.0 0.0 0.0
            0.0 -2.0 0.0 0.0
            0.0 0.0 0.0 0.0
            0.0 0.0 0.0 -2.0
        ]
        for singular_term in (nothing, singular_matrix)
            prob = BVProblem(
                packed_test_matrix!, packed_test_bc!, [0.3 0.4; 0.7 0.8],
                (0.0, 1.0), (; rate = [1.2]); singular_term
            )
            cache, device_cache = packed_test_cache(prob, MIRK4())
            u = copy(cache.y₀_flat)
            @test packed_test_residual(cache, device_cache, u, true) ≈
                packed_test_residual(cache, device_cache, u, false)
        end
        bad_prob = BVProblem(
            packed_test_f!, packed_test_bc!, [0.3, 0.7], (0.0, 1.0), (; rate = [1.2]);
            singular_term = ones(1, 1)
        )
        cache, device_cache = packed_test_cache(bad_prob, MIRK4())
        @test_throws DimensionMismatch packed_test_residual(cache, device_cache, copy(cache.y₀_flat), true)
    end

    @testset "Tuned vector parameters" begin
        function f!(du, u, p, t)
            du[1] = u[2]
            du[2] = -p[1] * sin(u[1])
            return nothing
        end
        function bc!(res, u, p, t)
            res[1] = u(0.0)[1]
            res[2] = u(0.0)[2] - 1
            res[3] = u(1.0)[1] - 1
            return nothing
        end
        prob = BVProblem(f!, bc!, [0.3, 0.7], (0.0, 1.0), [1.2]; tune_parameters = true)
        cache, device_cache = packed_test_cache(prob, MIRK4())
        u = copy(cache.y₀_flat)
        cpu_loss = x -> packed_test_residual(cache, device_cache, x, false)
        device_loss = x -> packed_test_residual(cache, device_cache, x, true)
        @test device_loss(u) ≈ cpu_loss(u)
        @test ForwardDiff.jacobian(device_loss, u) ≈ ForwardDiff.jacobian(cpu_loss, u)
    end

    @testset "State and control dimensions" begin
        # An f_prototype does not reduce the state equations in an unconstrained BVP.
        full_f = BVPFunction(packed_test_f!, packed_test_bc!; f_prototype = zeros(1))
        full_prob = BVProblem(full_f, [0.3, 0.7], (0.0, 1.0), (; rate = [1.2]))
        cache, device_cache = packed_test_cache(full_prob, MIRK4())
        u = copy(cache.y₀_flat)
        expected = packed_test_residual(cache, device_cache, u, false)
        @test length(expected) == cache.M * length(cache.mesh_dt)
        @test packed_test_residual(cache, device_cache, u, true) ≈ expected

        # Control variables are interpolated without RK increments.
        control_f!(du, u, p, t) = (du[1] = u[2])
        control_f = BVPFunction(control_f!, packed_test_bc!; f_prototype = zeros(1))
        control_prob = BVProblem(
            control_f, [0.3, 0.7], (0.0, 1.0); lb = [-10.0, -10.0], ub = [10.0, 10.0]
        )
        cache, device_cache = packed_test_cache(control_prob, MIRK4())
        u = copy(cache.y₀_flat)
        cpu_loss = x -> packed_test_residual(cache, device_cache, x, false; constraint = Val(true))
        device_loss = x -> packed_test_residual(cache, device_cache, x, true; constraint = Val(true))
        @test length(cpu_loss(u)) == length(cache.mesh_dt)
        @test device_loss(u) ≈ cpu_loss(u)
        @test ForwardDiff.jacobian(device_loss, u) ≈ ForwardDiff.jacobian(cpu_loss, u)
    end
end

# Run the offload path with CPU kernel execution.
struct HostDevice <: MIRK.Backend end
MIRK.KernelAbstractions.allocate(::HostDevice, ::Type{T}, dims::Tuple) where {T} =
    Array{T}(undef, dims)
MIRK.KernelAbstractions.synchronize(::HostDevice) = nothing
MIRK.__mirk_packed_collocation_kernel!(::HostDevice) =
    MIRK.__mirk_packed_collocation_kernel!(CPU())

include("device_backend_tests.jl")

@testset "Device backend integration on CPU" begin
    # All tableaus are checked directly above; MIRK4 covers the common solve path.
    test_device_backend(HostDevice(); algorithms = (MIRK4,))
end

# Exercise the resident kernels with their CPU sparse storage adapter. This keeps
# differentiation and mesh-rebuild coverage in Core, without requiring a GPU.
function sparse_resident_cache(prob, alg = MIRK4(); dt = 0.2, adaptive = false, abstol = 1.0e-9)
    return MIRK.__init_mirk_device(
        prob, alg, prob.u0; dt, abstol, adaptive, controller = DefectControl(),
        nlsolve_kwargs = (; abstol), optimize_kwargs = (; abstol), verbose = false
    )
end

sparse_plan(cache) = cache.jacobian_cache[]

function sparse_pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.81 * sin(u[1]) + t * u[2]^2
    return nothing
end
function sparse_interior_bc!(res, u, p, t)
    res[1] = u(0.13)[1] + u(0.37)[2]^2 - 0.2
    res[2] = sin(u(0.87)[1]) - u(t[end])[2] - 0.1
    return nothing
end
function sparse_derivative_bc!(res, u, p, t)
    res[1] = u(0.13, Val{1})[2] + u(t[1])[1] - 0.2
    res[2] = u(0.87, Val{1})[1] - u(t[end])[2] - 0.1
    return nothing
end
function sparse_left_bc!(res, u, p)
    res[1] = u[1]^2 + u[2] - 0.2
    return nothing
end
function sparse_right_bc!(res, u, p)
    res[1] = sin(u[1]) - u[2] - 0.1
    return nothing
end
function sparse_matrix_rhs!(du, u, p, t)
    du[1, 1] = u[2, 1]
    du[2, 1] = -sin(u[1, 1]) + u[1, 2]^2
    du[1, 2] = u[2, 2]
    du[2, 2] = u[1, 1] * u[2, 2] - sin(u[1, 2])
    return nothing
end
function sparse_matrix_bc!(res, u, p, t)
    a, b = u(0.13), u(0.87)
    res[1] = a[1, 1] - 0.2
    res[2] = a[1, 2]^2 + a[2, 1] - 0.1
    res[3] = b[1, 1] + b[2, 2] - 0.3
    res[4] = sin(b[1, 2]) - 0.4
    return nothing
end

function sparse_jacobian_problems()
    return (
        BVProblem(sparse_pendulum!, sparse_interior_bc!, [0.2, 0.3], (0.0, 1.0)),
        BVProblem(sparse_pendulum!, sparse_derivative_bc!, [0.2, 0.3], (0.0, 1.0)),
        TwoPointBVProblem(
            sparse_pendulum!, (sparse_left_bc!, sparse_right_bc!), [0.2, 0.3], (0.0, 1.0);
            bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
        ),
        BVProblem(sparse_matrix_rhs!, sparse_matrix_bc!, [0.2 0.4; 0.3 0.5], (0.0, 1.0)),
    )
end

function sparse_dense_reference(cache, x)
    return ForwardDiff.jacobian(x) do state
        residual = similar(state, length(cache.residual[]))
        MIRK.__device_residual!(residual, state, cache)
        return residual
    end
end

function sparse_cpu_reference(prob, x; dt = 0.2)
    # Independently assemble the ordinary CPU residual, then differentiate one
    # coordinate at a time. This also detects a shared error in resident residuals.
    alg = MIRK4(; jac_alg = BVPJacobianAlgorithm(AutoFiniteDiff()))
    cache = init(prob, alg; dt, adaptive = false)
    nres = length(x)
    f = if prob.f.bc === sparse_derivative_bc!
        # The ordinary EvalSol derivative overload assumes DiffCache storage.
        # Use the CPU solution interpolation, which also supports plain arrays.
        function (state)
            y = MIRK.recursive_unflatten!(cache.y, state)
            MIRK.Φ!(
                cache.residual[2:end], cache, y, state,
                MIRK.NoDiffCacheNeeded(), Val(false)
            )
            MIRK.interp_setup!(cache)
            interp = MIRK.MIRKInterpolation(cache.mesh, y, cache)
            evaluate = function (t, derivative = Val{0})
                # Like EvalSol, value queries exactly on the mesh return the
                # unknown at that node even before the collocation residual is zero.
                node = findfirst(==(t), cache.mesh)
                derivative === Val{0} && node !== nothing && return y[node]
                return interp(t, nothing, derivative, cache.p)
            end
            prob.f.bc(cache.residual[1], evaluate, cache.p, cache.mesh)
            return reduce(vcat, cache.residual)
        end
    else
        nlprob = MIRK.__construct_problem(cache, copy(cache.y₀_flat), cache.y₀)
        function (state)
            residual = similar(state, nres)
            nlprob.f(residual, state, nlprob.p)
            return residual
        end
    end
    jacobian = Matrix{eltype(x)}(undef, nres, length(x))
    for col in eachindex(x)
        h = cbrt(eps(eltype(x))) * max(abs(x[col]), one(eltype(x)))
        plus, minus = copy(x), copy(x)
        plus[col] += h
        minus[col] -= h
        jacobian[:, col] = (f(plus) - f(minus)) / (2h)
    end
    return f(x), jacobian
end

@testset "Resident sparse differentiation matches independent derivatives" begin
    modes = (
        BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(; chunksize = 1))),
        BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(; chunksize = 2))),
        BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(; chunksize = 8))),
        BVPJacobianAlgorithm(AutoSparse(AutoFiniteDiff())),
        BVPJacobianAlgorithm(AutoSparse(AutoFiniteDiff(; fdjtype = Val(:central)))),
        BVPJacobianAlgorithm(;
            bc_diffmode = AutoFiniteDiff(), nonbc_diffmode = AutoForwardDiff(; chunksize = 2)
        ),
        BVPJacobianAlgorithm(;
            bc_diffmode = AutoForwardDiff(; chunksize = 1),
            nonbc_diffmode = AutoFiniteDiff(; fdjtype = Val(:central))
        ),
    )
    @testset "problem $index" for (index, prob) in enumerate(sparse_jacobian_problems())
        reference_cache = sparse_resident_cache(prob)
        x = vec(copy(reference_cache.y[]))
        x .+= 0.2 .* sin.(eachindex(x))
        expected = sparse_dense_reference(reference_cache, x)
        cpu_residual, cpu_jacobian = sparse_cpu_reference(prob, x)
        residual = similar(reference_cache.residual[])
        MIRK.__device_residual!(residual, x, reference_cache)
        @test residual ≈ cpu_residual rtol = 1.0e-11 atol = 1.0e-11
        @test expected ≈ cpu_jacobian rtol = 1.0e-7 atol = 1.0e-8

        @testset "mode $mode" for mode in modes
            cache = sparse_resident_cache(prob, MIRK4(; jac_alg = mode))
            J = cache.jac_prototype[]
            @test J isa SparseMatrixCSC
            @test !sparse_plan(cache).boundary_fallback
            @test MIRK.__device_jacobian!(J, x, cache) === J
            @test Matrix(J) ≈ expected rtol = 2.0e-6 atol = 1.0e-7

            # Reuse the same structure at a new nonlinear state: stored entries
            # must be overwritten, and compressed seeds must not alias columns.
            changed = x .+ 0.1 .* cos.(eachindex(x))
            expected_changed = sparse_dense_reference(reference_cache, changed)
            MIRK.__device_jacobian!(J, changed, cache)
            @test Matrix(J) ≈ expected_changed rtol = 2.0e-6 atol = 1.0e-7
        end
    end
end

@testset "Sparse differentiation across MIRK tableaus" begin
    prob = first(sparse_jacobian_problems())
    @testset "$Alg" for Alg in (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6, MIRK6I)
        cache = sparse_resident_cache(prob, Alg())
        x = vec(copy(cache.y[])) .+ 0.1 .* sin.(1:length(cache.y[]))
        expected = sparse_dense_reference(cache, x)
        MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
        @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11
    end
end

function sparse_branch_bc!(res, u, p, t)
    if u(t[1])[1] > 0
        res[1] = u(0.13)[1]^2
    else
        res[1] = u(0.87)[2]^2
    end
    res[2] = u(t[end])[1] - 0.1
    return nothing
end

function sparse_global_bc!(res, u, p, t)
    res[1] = zero(eltype(res))
    for index in eachindex(t)
        res[1] += sum(u[index])
    end
    res[2] = u(t[end])[1] - 0.1
    return nothing
end

@testset "Conservative boundary fallback and explicit boundary sparsity" begin
    prob = BVProblem(sparse_pendulum!, sparse_branch_bc!, [0.2, 0.3], (0.0, 1.0))
    cache = sparse_resident_cache(prob)
    plan = sparse_plan(cache)
    @test plan.boundary_fallback
    @test nnz(plan.pattern[1:2, :]) == 2length(cache.y[])
    @test maximum(group.ncolors for group in plan.groups if !group.boundary) == 2cache.M
    # The initialized pattern must remain valid when a later iterate selects a
    # branch depending on entirely different mesh intervals.
    for sign in (-1, 1)
        x = sign .* vec(copy(cache.y[]))
        expected = sparse_dense_reference(cache, x)
        MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
        @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11
    end

    global_prob = BVProblem(sparse_pendulum!, sparse_global_bc!, [0.2, 0.3], (0.0, 1.0))
    global_cache = sparse_resident_cache(global_prob)
    global_plan = sparse_plan(global_cache)
    @test !global_plan.boundary_fallback
    @test any(group.boundary && group.ncolors == length(global_cache.y[]) for group in global_plan.groups)
    @test any(!group.boundary && group.ncolors == 2global_cache.M for group in global_plan.groups)
    x = vec(copy(global_cache.y[]))
    expected = sparse_dense_reference(global_cache, x)
    MIRK.__device_jacobian!(global_cache.jac_prototype[], x, global_cache)
    @test Matrix(global_cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11

    prob = first(sparse_jacobian_problems())
    reference = sparse_resident_cache(prob)
    known_pattern = copy(sparse_plan(reference).pattern[1:2, :])
    fill!(nonzeros(known_pattern), 0) # Explicit zeros still specify structural entries.
    jac_alg = BVPJacobianAlgorithm(;
        bc_diffmode = AutoSparse(
            AutoForwardDiff(); sparsity_detector = KnownJacobianSparsityDetector(known_pattern)
        ),
        nonbc_diffmode = AutoForwardDiff()
    )
    known = sparse_resident_cache(prob, MIRK4(; jac_alg))
    @test !sparse_plan(known).boundary_fallback
    @test nnz(sparse_plan(known).pattern[1:2, :]) == nnz(known_pattern)
    x = vec(copy(known.y[]))
    expected = sparse_dense_reference(reference, x)
    MIRK.__device_jacobian!(known.jac_prototype[], x, known)
    @test Matrix(known.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11

    invalid_mode = AutoSparse(
        AutoForwardDiff();
        sparsity_detector = KnownJacobianSparsityDetector(zeros(3, length(known.y[])))
    )
    invalid_alg = MIRK4(; jac_alg = BVPJacobianAlgorithm(; bc_diffmode = invalid_mode))
    @test_throws DimensionMismatch sparse_resident_cache(prob, invalid_alg)
end

@testset "Float32 sparse differentiation" begin
    prob = BVProblem(sparse_pendulum!, sparse_interior_bc!, Float32[0.2, 0.3], (0.0f0, 1.0f0))
    for mode in (
            AutoForwardDiff(; chunksize = 2), AutoFiniteDiff(),
            AutoFiniteDiff(; fdjtype = Val(:central)),
        )
        cache = sparse_resident_cache(prob, MIRK4(; jac_alg = BVPJacobianAlgorithm(mode)); dt = 0.2f0)
        x = vec(copy(cache.y[]))
        expected = sparse_dense_reference(cache, x)
        MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
        @test eltype(cache.jac_prototype[]) === Float32
        @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 3.0f-3 atol = 5.0f-4
    end
end

@testset "Sparse storage and color count are bounded with mesh growth" begin
    prob = BVProblem(sparse_pendulum!, sparse_interior_bc!, [0.2, 0.3], (0.0, pi / 2))
    for dt in (0.02, 0.001)
        cache = sparse_resident_cache(prob; dt)
        plan = sparse_plan(cache)
        N = length(cache.mesh_dt[])
        M = cache.M
        @test cache.jac_prototype[] isa SparseMatrixCSC
        # Each collocation block touches two states. Each test boundary row
        # touches at most three nearby states, independent of the mesh size.
        @test nnz(cache.jac_prototype[]) <= 2M^2 * N + 12M
        @test nnz(cache.jac_prototype[]) < length(cache.jac_prototype[]) ÷ 10
        @test all(group.ncolors <= 6M for group in plan.groups)
        @test sum(cld(group.ncolors, 8) for group in plan.groups) <= 4
        @test !plan.boundary_fallback
    end
end

@testset "Mesh refinement rebuilds the sparse derivative plan" begin
    prob = first(sparse_jacobian_problems())
    cache = sparse_resident_cache(prob)
    old_J, old_plan = cache.jac_prototype[], sparse_plan(cache)
    old_nodes = length(cache.mesh[])
    MIRK.__device_residual!(cache.residual[], vec(cache.y[]), cache)
    MIRK.half_mesh!(cache)
    @test length(cache.mesh[]) == 2old_nodes - 1
    @test cache.jac_prototype[] isa SparseMatrixCSC
    @test cache.jac_prototype[] !== old_J
    @test sparse_plan(cache) !== old_plan
    @test size(cache.jac_prototype[]) == (length(cache.residual[]), length(cache.y[]))
    x = vec(copy(cache.y[]))
    expected = sparse_dense_reference(cache, x)
    MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
    @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11
end

function sparse_solvable_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = 2 * u[1]^3
    return nothing
end
function sparse_solvable_bc!(res, u, p, t)
    res[1] = u(0.13)[1] - 1 / (2 - 0.13)
    res[2] = u(0.87)[1] - 1 / (2 - 0.87)
    return nothing
end
function sparse_parameter_bc!(res, u, p, t)
    res[1] = u(p[1])[1] - 1 / (2 - p[1])
    res[2] = u(p[2])[1] - 1 / (2 - p[2])
    return nothing
end

@testset "Resident sparse nonlinear solve" begin
    prob = BVProblem(sparse_solvable_rhs!, sparse_solvable_bc!, [0.75, 0.5], (0.0, 1.0))
    cache = sparse_resident_cache(prob; dt = 0.1)
    cpu = solve(prob, MIRK4(); dt = 0.1, adaptive = false, abstol = 1.0e-9)
    sol = solve!(cache)
    @test successful_retcode(cpu)
    @test successful_retcode(sol)
    @test cache.jac_prototype[] isa SparseMatrixCSC
    @test sol.original.prob.f.jac_prototype isa SparseMatrixCSC
    @test Array(sol) ≈ Array(cpu) rtol = 1.0e-8 atol = 1.0e-9
    @test sol(0.43) ≈ [1 / (2 - 0.43), 1 / (2 - 0.43)^2] rtol = 1.0e-4

    adaptive_cache = sparse_resident_cache(prob; dt = 0.25, adaptive = true, abstol = 1.0e-7)
    adaptive_sol = solve!(adaptive_cache)
    @test successful_retcode(adaptive_sol)
    @test length(adaptive_sol.t) > 5
    @test adaptive_cache.jac_prototype[] isa SparseMatrixCSC
    @test size(adaptive_cache.jac_prototype[], 2) == 2length(adaptive_sol.t)
    @test adaptive_sol(0.43) ≈ [1 / (2 - 0.43), 1 / (2 - 0.43)^2] rtol = 1.0e-5
end

@testset "Reused sparse solve updates parameter-dependent boundary structure" begin
    prob = BVProblem(
        sparse_solvable_rhs!, sparse_parameter_bc!, [0.75, 0.5], (0.0, 1.0), [0.13, 0.87]
    )
    cache = sparse_resident_cache(prob; dt = 0.1)
    initial = solve!(cache)
    @test successful_retcode(initial)
    old_pattern = sparse_plan(cache).pattern
    cache.p .= [0.31, 0.69]
    changed = solve!(cache)
    @test successful_retcode(changed)
    @test sparse_plan(cache).pattern != old_pattern
    @test changed(0.43) ≈ [1 / (2 - 0.43), 1 / (2 - 0.43)^2] rtol = 1.0e-4
    x = vec(copy(cache.y[]))
    expected = sparse_dense_reference(cache, x)
    MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
    @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11
end

function sparse_nlls_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[1]
    return nothing
end
function sparse_nlls_bc!(res, u, p, t)
    res[1] = u(t[1])[1] - 1
    res[2] = u(t[end])[1] - exp(t[end])
    res[3] = u(t[1])[2] - 1
    return nothing
end

@testset "Resident sparse rectangular least-squares solve" begin
    bf = BVPFunction(sparse_nlls_rhs!, sparse_nlls_bc!; bcresid_prototype = zeros(3))
    prob = BVProblem(bf, [0.5, 0.5], (0.0, 1.0); nlls = Val(true))
    cache = sparse_resident_cache(prob; dt = 0.1, abstol = 1.0e-8)
    x = vec(copy(cache.y[]))
    expected = sparse_dense_reference(cache, x)
    MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
    @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11
    nlprob = MIRK.__construct_problem(cache, x)
    direction = sin.(eachindex(x))
    residual_direction = cos.(eachindex(cache.residual[]))
    jvp, vjp = similar(cache.residual[]), similar(x)
    nlprob.f.jvp(jvp, direction, x, nlprob.p)
    nlprob.f.vjp(vjp, residual_direction, x, nlprob.p)
    @test jvp ≈ expected * direction rtol = 1.0e-11 atol = 1.0e-11
    @test vjp ≈ expected' * residual_direction rtol = 1.0e-11 atol = 1.0e-11
    product_J = sparse_plan(cache).product
    @test product_J isa SparseMatrixCSC
    @test product_J !== cache.jac_prototype[]
    sol = solve!(cache)
    @test successful_retcode(sol)
    @test cache.jac_prototype[] isa SparseMatrixCSC
    @test size(cache.jac_prototype[]) == (length(x) + 1, length(x))
    @test sol.original.prob.f.jac_prototype isa SparseMatrixCSC
    @test sol.u[end] ≈ fill(exp(1.0), 2) rtol = 1.0e-4
end


function sparse_coupling_bc!(res, u, p, t)
    if iszero(p[1])
        res[1] = u(t[1])[1] - 0.5
    else
        res[1] = zero(eltype(res))
        for index in eachindex(t)
            exact = 1 / (2 - t[index])
            res[1] += u[index][1] + u[index][2] - exact - exact^2
        end
    end
    res[2] = u(t[end])[1] - 1
    return nothing
end

@testset "Reused sparse cache changes its coloring group count" begin
    prob = BVProblem(
        sparse_solvable_rhs!, sparse_coupling_bc!, [0.75, 0.5], (0.0, 1.0), [0.0]
    )
    cache = sparse_resident_cache(prob; dt = 0.1)
    initial_plan = sparse_plan(cache)
    @test length(initial_plan.groups) == 1
    @test !initial_plan.boundary_fallback
    initial_sol = solve!(cache)
    @test successful_retcode(initial_sol)

    for (coupling, ngroups) in ((1.0, 2), (0.0, 1))
        old_plan = sparse_plan(cache)
        cache.p[1] = coupling
        sol = solve!(cache)
        plan = sparse_plan(cache)
        @test successful_retcode(sol)
        @test plan !== old_plan
        @test typeof(plan) === typeof(initial_plan)
        @test length(plan.groups) == ngroups
        @test !plan.boundary_fallback
        @test sol(0.43) ≈ [1 / (2 - 0.43), 1 / (2 - 0.43)^2] rtol = 1.0e-4
        x = vec(copy(cache.y[]))
        expected = sparse_dense_reference(cache, x)
        MIRK.__device_jacobian!(cache.jac_prototype[], x, cache)
        @test Matrix(cache.jac_prototype[]) ≈ expected rtol = 1.0e-11 atol = 1.0e-11
    end
end
