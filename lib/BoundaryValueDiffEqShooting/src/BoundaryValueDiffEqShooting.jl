module BoundaryValueDiffEqShooting

using ADTypes
using ArrayInterface: fast_scalar_indexing
using BandedMatrices: BandedMatrix, Ones
using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
    recursive_flatten, recursive_flatten!, recursive_unflatten!,
    __concrete_solve_algorithm, diff!, __any_sparse_ad,
    __cache_trait, concrete_jacobian_algorithm, eval_bc_residual,
    eval_bc_residual!, get_tmp, __maybe_matmul!,
    __concrete_kwargs, __extract_problem_details,
    __initial_guess, __construct_internal_problem,
    __default_coloring_algorithm, __default_sparsity_detector,
    __maybe_allocate_diffcache, __get_bcresid_prototype,
    safe_similar, __vec, __vec_f, __vec_f!, __vec_bc, __vec_bc!,
    __materialize_jacobian_algorithm, __default_nonsparse_ad,
    recursive_flatten_twopoint!, __internal_nlsolve_problem,
    NoDiffCacheNeeded, DiffCacheNeeded, __extract_mesh,
    __extract_u0, __has_initial_guess, __initial_guess_length,
    __initial_guess_on_mesh, __flatten_initial_guess,
    __get_non_sparse_ad, __build_solution, __Fix3, get_dense_ad,
    __internal_solve

using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using DifferentiationInterface: DifferentiationInterface, Constant, prepare_jacobian,
    overloaded_input_type
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using LinearAlgebra
using Reexport: @reexport
using RecursiveArrayTools: ArrayPartition, DiffEqArray, VectorOfArray
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, StandardBVProblem, __solve,
    _unwrap_val, NonlinearProblem, NonlinearLeastSquaresProblem, OptimizationProblem
using Setfield: @set!, @set
using SparseArrays: sparse
using OrdinaryDiffEqTsit5: Tsit5
using PrecompileTools: @compile_workload, @setup_workload
using Preferences: Preferences

const DI = DifferentiationInterface

@reexport using ADTypes, BoundaryValueDiffEqCore, SciMLBase

include("algorithms.jl")
include("single_shooting.jl")
include("multiple_shooting.jl")
include("sparse_jacobians.jl")

@setup_workload begin
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end
    f1 = (u, p, t) -> [u[2], 0]

    function bc1!(residual, u, p, t)
        residual[1] = u(0.0)[1] - 5
        residual[2] = u(5.0)[1]
    end

    bc1 = (u, p, t) -> [u(0.0)[1] - 5, u(5.0)[1]]

    bc1_a! = (residual, ua, p) -> (residual[1] = ua[1] - 5)
    bc1_b! = (residual, ub, p) -> (residual[1] = ub[1])

    bc1_a = (ua, p) -> [ua[1] - 5]
    bc1_b = (ub, p) -> [ub[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [
        BVProblem(f1!, bc1!, u0, tspan; nlls = Val(false)),
        BVProblem(f1, bc1, u0, tspan; nlls = Val(false)),
        TwoPointBVProblem(
            f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype, nlls = Val(false)
        ),
        TwoPointBVProblem(
            f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype, nlls = Val(false)
        ),
    ]

    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    algs = []

    if Preferences.@load_preference("PrecompileShooting", true)
        append!(algs, [Shooting(Tsit5(); jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileMultipleShooting", true)
        append!(algs, [MultipleShooting(5, Tsit5(); jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg)
        end
    end
end

export Shooting, MultipleShooting

end # module BoundaryValueDiffEqShooting
