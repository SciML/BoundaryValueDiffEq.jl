module BoundaryValueDiffEqFIRK

using PrecompileTools: @compile_workload, @setup_workload
using SparseMatrixColorings: ColoringProblem, GreedyColoringAlgorithm,
                             ConstantColoringAlgorithm, row_colors, column_colors, coloring,
                             LargestFirst
using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using BandedMatrices: BandedMatrix, Ones
using SparseArrays: sparse
using LinearAlgebra
using FastAlmostBandedMatrices: AlmostBandedMatrix, fillpart, exclusive_bandpart,
                                finish_part_setindex!

using BoundaryValueDiffEqCore: BoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
                               recursive_flatten, recursive_flatten!, recursive_unflatten!,
                               __concrete_nonlinearsolve_algorithm, diff!,
                               __FastShortcutBVPCompatibleNonlinearPolyalg,
                               __FastShortcutBVPCompatibleNLLSPolyalg,
                               concrete_jacobian_algorithm, eval_bc_residual,
                               eval_bc_residual!, get_tmp, __maybe_matmul!,
                               __append_similar!, __extract_problem_details,
                               __initial_guess, __maybe_allocate_diffcache,
                               __get_bcresid_prototype, __similar, __vec, __vec_f, __vec_f!,
                               __vec_bc, __vec_bc!, recursive_flatten_twopoint!,
                               __internal_nlsolve_problem, MaybeDiffCache, __extract_mesh,
                               __extract_u0, __has_initial_guess, __initial_guess_length,
                               __initial_guess_on_mesh, __flatten_initial_guess,
                               __build_solution, __Fix3, __sparse_jacobian_cache,
                               __sparsity_detection_alg, _sparse_like, ColoredMatrix,
                               get_dense_ad

using Adapt
using ADTypes: ADTypes
using ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using Logging: Logging
using RecursiveArrayTools: AbstractVectorOfArray, VectorOfArray, recursivecopy
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, StandardBVProblem, __solve,
                 _unwrap_val
using Preferences: Preferences
using Reexport: @reexport
using Setfield: @set!
const DI = DifferentiationInterface

@reexport using BoundaryValueDiffEqCore, SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("firk.jl")
include("adaptivity.jl")
include("alg_utils.jl")
include("collocation.jl")
include("interpolation.jl")
include("lobatto_tableaus.jl")
include("radau_tableaus.jl")
include("sparse_jacobians.jl")

@setup_workload begin
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end
    f1 = (u, p, t) -> [u[2], 0]

    function bc1!(residual, u, p, t)
        residual[1] = u[:, 1][1] - 5
        residual[2] = u[:, end][1]
    end

    bc1 = (u, p, t) -> [u[:, 1][1] - 5, u[:, end][1]]

    bc1_a! = (residual, ua, p) -> (residual[1] = ua[1] - 5)
    bc1_b! = (residual, ub, p) -> (residual[1] = ub[1])

    bc1_a = (ua, p) -> [ua[1] - 5]
    bc1_b = (ub, p) -> [ub[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [BVProblem(f1!, bc1!, u0, tspan; nlls = Val(false)),
        BVProblem(f1, bc1, u0, tspan; nlls = Val(false)),
        TwoPointBVProblem(
            f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype, nlls = Val(false)),
        TwoPointBVProblem(
            f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype, nlls = Val(false))]

    algs = []

    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    if Preferences.@load_preference("PrecompileRadauIIa", true)
        append!(algs, [RadauIIa5(; jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileLobattoIIIa", true)
        append!(algs, [LobattoIIIa5(; jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileLobattoIIIb", true)
        append!(algs, [LobattoIIIb5(; jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileLobattoIIIc", true)
        append!(algs, [LobattoIIIc5(; jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2)
        end
    end

    #### NLLS precompile workload ####

    f1_nlls! = (du, u, p, t) -> begin
        du[1] = u[2]
        du[2] = -u[1]
    end

    f1_nlls = (u, p, t) -> [u[2], -u[1]]

    bc1_nlls! = (resid, sol, p, t) -> begin
        solₜ₁ = sol[:, 1]
        solₜ₂ = sol[:, end]
        resid[1] = solₜ₁[1]
        resid[2] = solₜ₂[1] - 1
        resid[3] = solₜ₂[2] + 1.729109
        return nothing
    end
    bc1_nlls = (sol, p, t) -> [sol[:, 1][1], sol[:, end][1] - 1, sol[:, end][2] + 1.729109]

    bc1_nlls_a! = (resid, ua, p) -> (resid[1] = ua[1])
    bc1_nlls_b! = (resid, ub, p) -> (resid[1] = ub[1] - 1;
    resid[2] = ub[2] + 1.729109)

    bc1_nlls_a = (ua, p) -> [ua[1]]
    bc1_nlls_b = (ub, p) -> [ub[1] - 1, ub[2] + 1.729109]

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0]
    bcresid_prototype1 = Array{Float64}(undef, 3)
    bcresid_prototype2 = (Array{Float64}(undef, 1), Array{Float64}(undef, 2))

    probs = [
        BVProblem(BVPFunction(f1_nlls!, bc1_nlls!; bcresid_prototype = bcresid_prototype1),
            u0, tspan, nlls = Val(true)),
        BVProblem(BVPFunction(f1_nlls, bc1_nlls; bcresid_prototype = bcresid_prototype1),
            u0, tspan, nlls = Val(true)),
        TwoPointBVProblem(f1_nlls!, (bc1_nlls_a!, bc1_nlls_b!), u0, tspan;
            bcresid_prototype = bcresid_prototype2, nlls = Val(true)),
        TwoPointBVProblem(f1_nlls, (bc1_nlls_a, bc1_nlls_b), u0, tspan;
            bcresid_prototype = bcresid_prototype2, nlls = Val(true))]

    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    nlsolvers = [LevenbergMarquardt(; disable_geodesic = Val(true)), GaussNewton()]

    algs = []

    if Preferences.@load_preference("PrecompileMIRKNLLS", false)
        for nlsolve in nlsolvers
            append!(algs, [MIRK2(; jac_alg, nlsolve), MIRK6(; jac_alg, nlsolve)])
        end
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2, abstol = 1e-2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileRadauIINLLS", false)
        for nlsolve in nlsolvers
            append!(algs,
                [RadauIIa2(; jac_alg, nlsolve), RadauIIa3(; jac_alg, nlsolve),
                    RadauIIa5(; jac_alg, nlsolve), RadauIIa7(; jac_alg, nlsolve)])
        end
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2, abstol = 1e-2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileLobattoIIIaNLLS", false)
        for nlsolve in nlsolvers
            append!(algs,
                [LobattoIIIa3(; jac_alg, nlsolve), LobattoIIIa4(; jac_alg, nlsolve),
                    LobattoIIIa5(; jac_alg, nlsolve)])
        end
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2, abstol = 1e-2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileLobattoIIIbNLLS", false)
        for nlsolve in nlsolvers
            append!(algs,
                [LobattoIIIb3(; jac_alg, nlsolve), LobattoIIIb4(; jac_alg, nlsolve),
                    LobattoIIIb5(; jac_alg, nlsolve)])
        end
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2, abstol = 1e-2)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileLobattoIIIcNLLS", false)
        for nlsolve in nlsolvers
            append!(algs,
                [LobattoIIIc3(; jac_alg, nlsolve), LobattoIIIc4(; jac_alg, nlsolve),
                    LobattoIIIc5(; jac_alg, nlsolve)])
        end
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg; dt = 0.2, abstol = 1e-2)
        end
    end
end

export RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7
export LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5
export BVPJacobianAlgorithm

end
