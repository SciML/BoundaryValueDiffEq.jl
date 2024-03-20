module BoundaryValueDiffEq

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes, Adapt, DiffEqBase, ForwardDiff, LinearAlgebra, NonlinearSolve,
          OrdinaryDiffEq, PreallocationTools, Preferences, RecursiveArrayTools, Reexport,
          SciMLBase, Setfield, SparseDiffTools

    # Special Matrix Types
    using BandedMatrices, FastAlmostBandedMatrices, SparseArrays

    import ADTypes: AbstractADType
    import ArrayInterface: matrix_colors, parameterless_type, undefmatrix,
                           fast_scalar_indexing
    import ConcreteStructs: @concrete
    import DiffEqBase: solve
    import FastClosures: @closure
    import ForwardDiff: pickchunksize
    import RecursiveArrayTools: ArrayPartition, DiffEqArray
    import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val
    import SparseDiffTools: AbstractSparseADType
    import UnPack: @unpack
end

@reexport using ADTypes, DiffEqBase, NonlinearSolve, OrdinaryDiffEq, SparseDiffTools,
                SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")

include("mirk_tableaus.jl")

include("solve/single_shooting.jl")
include("solve/multiple_shooting.jl")
include("solve/mirk.jl")

include("collocation.jl")
include("sparse_jacobians.jl")

include("adaptivity.jl")
include("interpolation.jl")

function __solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

@setup_workload begin
    # function f1!(du, u, p, t)
    #     du[1] = u[2]
    #     du[2] = 0
    # end
    # f1(u, p, t) = [u[2], 0]

    # function bc1!(residual, u, p, t)
    #     residual[1] = u[1][1] - 5
    #     residual[2] = u[end][1]
    # end
    # bc1(u, p, t) = [u[1][1] - 5, u[end][1]]

    # bc1_a!(residual, ua, p) = (residual[1] = ua[1] - 5)
    # bc1_b!(residual, ub, p) = (residual[1] = ub[1])

    # bc1_a(ua, p) = [ua[1] - 5]
    # bc1_b(ub, p) = [ub[1]]

    # tspan = (0.0, 5.0)
    # u0 = [5.0, -3.5]
    # bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    # probs = [BVProblem(f1!, bc1!, u0, tspan), BVProblem(f1, bc1, u0, tspan),
    #     TwoPointBVProblem(f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype),
    #     TwoPointBVProblem(f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype)]

    # algs = []

    # jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    # if Preferences.@load_preference("PrecompileMIRK", true)
    #     append!(algs,
    #         [MIRK2(; jac_alg), MIRK3(; jac_alg), MIRK4(; jac_alg),
    #             MIRK5(; jac_alg), MIRK6(; jac_alg)])
    # end

    # @compile_workload begin
    #     for prob in probs, alg in algs
    #         solve(prob, alg; dt = 0.2)
    #     end
    # end

    # function f1_nlls!(du, u, p, t)
    #     du[1] = u[2]
    #     du[2] = -u[1]
    # end

    # f1_nlls(u, p, t) = [u[2], -u[1]]

    # function bc1_nlls!(resid, sol, p, t)
    #     solₜ₁ = sol[1]
    #     solₜ₂ = sol[end]
    #     resid[1] = solₜ₁[1]
    #     resid[2] = solₜ₂[1] - 1
    #     resid[3] = solₜ₂[2] + 1.729109
    #     return nothing
    # end
    # bc1_nlls(sol, p, t) = [sol[1][1], sol[end][1] - 1, sol[end][2] + 1.729109]

    # bc1_nlls_a!(resid, ua, p) = (resid[1] = ua[1])
    # bc1_nlls_b!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)

    # bc1_nlls_a(ua, p) = [ua[1]]
    # bc1_nlls_b(ub, p) = [ub[1] - 1, ub[2] + 1.729109]

    # tspan = (0.0, 100.0)
    # u0 = [0.0, 1.0]
    # bcresid_prototype1 = Array{Float64}(undef, 3)
    # bcresid_prototype2 = (Array{Float64}(undef, 1), Array{Float64}(undef, 2))

    # probs = [
    #     BVProblem(BVPFunction(f1_nlls!, bc1_nlls!; bcresid_prototype = bcresid_prototype1),
    #         u0, tspan),
    #     BVProblem(BVPFunction(f1_nlls, bc1_nlls; bcresid_prototype = bcresid_prototype1),
    #         u0, tspan),
    #     TwoPointBVProblem(f1_nlls!, (bc1_nlls_a!, bc1_nlls_b!), u0,
    #         tspan; bcresid_prototype = bcresid_prototype2),
    #     TwoPointBVProblem(f1_nlls, (bc1_nlls_a, bc1_nlls_b), u0, tspan;
    #         bcresid_prototype = bcresid_prototype2)]

    # jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    # nlsolvers = [LevenbergMarquardt(), GaussNewton()]

    # algs = []

    # if Preferences.@load_preference("PrecompileMIRKNLLS", false)
    #     for nlsolve in nlsolvers
    #         append!(algs,
    #             [MIRK2(; jac_alg, nlsolve), MIRK3(; jac_alg, nlsolve),
    #                 MIRK4(; jac_alg, nlsolve), MIRK5(; jac_alg, nlsolve),
    #                 MIRK6(; jac_alg, nlsolve)])
    #     end
    # end

    # @compile_workload begin
    #     for prob in probs, alg in algs
    #         solve(prob, alg; dt = 0.2)
    #     end
    # end

    # function f1!(du, u, p, t)
    #     du[1] = u[2]
    #     du[2] = 0
    # end
    # f1(u, p, t) = [u[2], 0]

    # function bc1!(residual, u, p, t)
    #     residual[1] = u(0.0)[1] - 5
    #     residual[2] = u(5.0)[1]
    # end
    # bc1(u, p, t) = [u(0.0)[1] - 5, u(5.0)[1]]

    # bc1_a!(residual, ua, p) = (residual[1] = ua[1] - 5)
    # bc1_b!(residual, ub, p) = (residual[1] = ub[1])

    # bc1_a(ua, p) = [ua[1] - 5]
    # bc1_b(ub, p) = [ub[1]]

    # tspan = (0.0, 5.0)
    # u0 = [5.0, -3.5]
    # bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    # probs = [BVProblem(BVPFunction{true}(f1!, bc1!), u0, tspan; nlls = Val(false)),
    #     BVProblem(BVPFunction{false}(f1, bc1), u0, tspan; nlls = Val(false)),
    #     BVProblem(
    #         BVPFunction{true}(
    #             f1!, (bc1_a!, bc1_b!); bcresid_prototype, twopoint = Val(true)),
    #         u0,
    #         tspan;
    #         nlls = Val(false)),
    #     BVProblem(
    #         BVPFunction{false}(f1, (bc1_a, bc1_b); bcresid_prototype, twopoint = Val(true)),
    #         u0, tspan; nlls = Val(false))]

    # algs = []

    # if @load_preference("PrecompileShooting", true)
    #     push!(algs,
    #         Shooting(Tsit5(); nlsolve = NewtonRaphson(),
    #             jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))))
    # end

    # if @load_preference("PrecompileMultipleShooting", true)
    #     push!(algs,
    #         MultipleShooting(10,
    #             Tsit5();
    #             nlsolve = NewtonRaphson(),
    #             jac_alg = BVPJacobianAlgorithm(;
    #                 bc_diffmode = AutoForwardDiff(; chunksize = 2),
    #                 nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))))
    # end

    # @compile_workload begin
    #     for prob in probs, alg in algs
    #         solve(prob, alg)
    #     end
    # end

    # function f1_nlls!(du, u, p, t)
    #     du[1] = u[2]
    #     du[2] = -u[1]
    # end

    # f1_nlls(u, p, t) = [u[2], -u[1]]

    # function bc1_nlls!(resid, sol, p, t)
    #     solₜ₁ = sol(0.0)
    #     solₜ₂ = sol(100.0)
    #     resid[1] = solₜ₁[1]
    #     resid[2] = solₜ₂[1] - 1
    #     resid[3] = solₜ₂[2] + 1.729109
    #     return nothing
    # end
    # bc1_nlls(sol, p, t) = [sol(0.0)[1], sol(100.0)[1] - 1, sol(1.0)[2] + 1.729109]

    # bc1_nlls_a!(resid, ua, p) = (resid[1] = ua[1])
    # bc1_nlls_b!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)

    # bc1_nlls_a(ua, p) = [ua[1]]
    # bc1_nlls_b(ub, p) = [ub[1] - 1, ub[2] + 1.729109]

    # tspan = (0.0, 100.0)
    # u0 = [0.0, 1.0]
    # bcresid_prototype1 = Array{Float64}(undef, 3)
    # bcresid_prototype2 = (Array{Float64}(undef, 1), Array{Float64}(undef, 2))

    # probs = [
    #     BVProblem(
    #         BVPFunction{true}(f1_nlls!, bc1_nlls!; bcresid_prototype = bcresid_prototype1),
    #         u0, tspan; nlls = Val(true)),
    #     BVProblem(
    #         BVPFunction{false}(f1_nlls, bc1_nlls; bcresid_prototype = bcresid_prototype1),
    #         u0, tspan; nlls = Val(true)),
    #     BVProblem(
    #         BVPFunction{true}(f1_nlls!, (bc1_nlls_a!, bc1_nlls_b!);
    #             bcresid_prototype = bcresid_prototype2, twopoint = Val(true)),
    #         u0,
    #         tspan;
    #         nlls = Val(true)),
    #     BVProblem(
    #         BVPFunction{false}(f1_nlls, (bc1_nlls_a, bc1_nlls_b);
    #             bcresid_prototype = bcresid_prototype2, twopoint = Val(true)),
    #         u0,
    #         tspan;
    #         nlls = Val(true))]

    # algs = []

    # if @load_preference("PrecompileShootingNLLS", true)
    #     append!(algs,
    #         [
    #             Shooting(Tsit5(); nlsolve = TrustRegion(),
    #                 jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))),
    #             Shooting(Tsit5(); nlsolve = GaussNewton(),
    #                 jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2)))])
    # end

    # if @load_preference("PrecompileMultipleShootingNLLS", true)
    #     append!(algs,
    #         [
    #             MultipleShooting(10,
    #                 Tsit5();
    #                 nlsolve = TrustRegion(),
    #                 jac_alg = BVPJacobianAlgorithm(;
    #                     bc_diffmode = AutoForwardDiff(; chunksize = 2),
    #                     nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))),
    #             MultipleShooting(10,
    #                 Tsit5();
    #                 nlsolve = GaussNewton(),
    #                 jac_alg = BVPJacobianAlgorithm(;
    #                     bc_diffmode = AutoForwardDiff(; chunksize = 2),
    #                     nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2)))])
    # end

    # @compile_workload begin
    #     for prob in probs, alg in algs
    #         solve(prob, alg)
    #     end
    # end
end

export Shooting, MultipleShooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export BVPM2, BVPSOL, COLNEW # From ODEInterface.jl

export MIRKJacobianComputationAlgorithm, BVPJacobianAlgorithm

end
