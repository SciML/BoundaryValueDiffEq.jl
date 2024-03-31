module BoundaryValueDiffEq

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes, Adapt, DiffEqBase, ForwardDiff, LinearAlgebra, NonlinearSolve,
          OrdinaryDiffEq, Preferences, RecursiveArrayTools, Reexport, SciMLBase, Setfield,
          SparseDiffTools

    using PreallocationTools: PreallocationTools, DiffCache

    # Special Matrix Types
    using BandedMatrices, FastAlmostBandedMatrices, SparseArrays

    import ADTypes: AbstractADType
    import ArrayInterface: matrix_colors, parameterless_type, undefmatrix,
                           fast_scalar_indexing
    import ConcreteStructs: @concrete
    import DiffEqBase: solve
    import FastClosures: @closure
    import ForwardDiff: ForwardDiff, pickchunksize
    import Logging
    import RecursiveArrayTools: ArrayPartition, DiffEqArray
    import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val
    import SparseDiffTools: AbstractSparseADType
end

@reexport using ADTypes, DiffEqBase, NonlinearSolve, OrdinaryDiffEq, SparseDiffTools,
                SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")

include("mirk_tableaus.jl")
include("lobatto_tableaus.jl")
include("radau_tableaus.jl")


include("solve/single_shooting.jl")
include("solve/multiple_shooting.jl")
include("solve/firk.jl")
include("solve/mirk.jl")

include("collocation.jl")
include("sparse_jacobians.jl")

include("adaptivity.jl")
include("interpolation.jl")

include("default_nlsolve.jl")

function __solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

@setup_workload begin
    f1! = @closure (du, u, p, t) -> begin
        du[1] = u[2]
        du[2] = 0
    end
    f1 = @closure (u, p, t) -> [u[2], 0]

    bc1! = @closure (residual, u, p, t) -> begin
        residual[1] = u[1][1] - 5
        residual[2] = u[lastindex(u)][1]
    end

    bc1 = @closure (u, p, t) -> [u[1][1] - 5, u[lastindex(u)][1]]

    bc1_a! = @closure (residual, ua, p) -> (residual[1] = ua[1] - 5)
    bc1_b! = @closure (residual, ub, p) -> (residual[1] = ub[1])

    bc1_a = @closure (ua, p) -> [ua[1] - 5]
    bc1_b = @closure (ub, p) -> [ub[1]]

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

    if Preferences.@load_preference("PrecompileMIRK", true)
        append!(algs, [MIRK2(; jac_alg), MIRK4(; jac_alg), MIRK6(; jac_alg)])
    end

    @compile_workload begin
        for prob in probs, alg in algs
            solve(prob, alg; dt = 0.2)
        end
    end

    f1_nlls! = @closure (du, u, p, t) -> begin
        du[1] = u[2]
        du[2] = -u[1]
    end

    f1_nlls = @closure (u, p, t) -> [u[2], -u[1]]

    bc1_nlls! = @closure (resid, sol, p, t) -> begin
        solₜ₁ = sol[1]
        solₜ₂ = sol[lastindex(sol)]
        resid[1] = solₜ₁[1]
        resid[2] = solₜ₂[1] - 1
        resid[3] = solₜ₂[2] + 1.729109
        return nothing
    end
    bc1_nlls = @closure (sol, p, t) -> [
        sol[1][1], sol[lastindex(sol)][1] - 1, sol[lastindex(sol)][2] + 1.729109]

    bc1_nlls_a! = @closure (resid, ua, p) -> (resid[1] = ua[1])
    bc1_nlls_b! = @closure (resid, ub, p) -> (resid[1] = ub[1] - 1;
    resid[2] = ub[2] + 1.729109)

    bc1_nlls_a = @closure (ua, p) -> [ua[1]]
    bc1_nlls_b = @closure (ub, p) -> [ub[1] - 1, ub[2] + 1.729109]

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
        for prob in probs, alg in algs
            solve(prob, alg; dt = 0.2, abstol = 1e-2)
        end
    end

    bc1! = @closure (residual, u, p, t) -> begin
        residual[1] = u(0.0)[1] - 5
        residual[2] = u(5.0)[1]
    end
    bc1 = @closure (u, p, t) -> [u(0.0)[1] - 5, u(5.0)[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [BVProblem(BVPFunction{true}(f1!, bc1!), u0, tspan; nlls = Val(false)),
        BVProblem(BVPFunction{false}(f1, bc1), u0, tspan; nlls = Val(false)),
        BVProblem(
            BVPFunction{true}(
                f1!, (bc1_a!, bc1_b!); bcresid_prototype, twopoint = Val(true)),
            u0,
            tspan;
            nlls = Val(false)),
        BVProblem(
            BVPFunction{false}(f1, (bc1_a, bc1_b); bcresid_prototype, twopoint = Val(true)),
            u0, tspan; nlls = Val(false))]

    algs = []

    if @load_preference("PrecompileShooting", true)
        push!(algs,
            Shooting(Tsit5(); nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))))
    end

    if @load_preference("PrecompileMultipleShooting", true)
        push!(algs,
            MultipleShooting(10,
                Tsit5();
                nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm(;
                    bc_diffmode = AutoForwardDiff(; chunksize = 2),
                    nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))))
    end

    @compile_workload begin
        for prob in probs, alg in algs
            solve(prob, alg)
        end
    end

    bc1_nlls! = @closure (resid, sol, p, t) -> begin
        solₜ₁ = sol(0.0)
        solₜ₂ = sol(100.0)
        resid[1] = solₜ₁[1]
        resid[2] = solₜ₂[1] - 1
        resid[3] = solₜ₂[2] + 1.729109
        return nothing
    end
    bc1_nlls = @closure (sol, p, t) -> [
        sol(0.0)[1], sol(100.0)[1] - 1, sol(1.0)[2] + 1.729109]

    tspan = (0.0, 100.0)
    u0 = [0.0, 1.0]
    bcresid_prototype1 = Array{Float64}(undef, 3)
    bcresid_prototype2 = (Array{Float64}(undef, 1), Array{Float64}(undef, 2))

    probs = [
        BVProblem(
            BVPFunction{true}(f1_nlls!, bc1_nlls!; bcresid_prototype = bcresid_prototype1),
            u0, tspan; nlls = Val(true)),
        BVProblem(
            BVPFunction{false}(f1_nlls, bc1_nlls; bcresid_prototype = bcresid_prototype1),
            u0, tspan; nlls = Val(true)),
        BVProblem(
            BVPFunction{true}(f1_nlls!, (bc1_nlls_a!, bc1_nlls_b!);
                bcresid_prototype = bcresid_prototype2, twopoint = Val(true)),
            u0,
            tspan;
            nlls = Val(true)),
        BVProblem(
            BVPFunction{false}(f1_nlls, (bc1_nlls_a, bc1_nlls_b);
                bcresid_prototype = bcresid_prototype2, twopoint = Val(true)),
            u0,
            tspan;
            nlls = Val(true))]

    algs = []

    if @load_preference("PrecompileShootingNLLS", true)
        append!(algs,
            [
                Shooting(
                    Tsit5(); nlsolve = LevenbergMarquardt(; disable_geodesic = Val(true)),
                    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))),
                Shooting(Tsit5(); nlsolve = GaussNewton(),
                    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2)))])
    end

    if @load_preference("PrecompileMultipleShootingNLLS", true)
        append!(algs,
            [
                MultipleShooting(10,
                    Tsit5();
                    nlsolve = LevenbergMarquardt(; disable_geodesic = Val(true)),
                    jac_alg = BVPJacobianAlgorithm(;
                        bc_diffmode = AutoForwardDiff(; chunksize = 2),
                        nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2))),
                MultipleShooting(10,
                    Tsit5();
                    nlsolve = GaussNewton(),
                    jac_alg = BVPJacobianAlgorithm(;
                        bc_diffmode = AutoForwardDiff(; chunksize = 2),
                        nonbc_diffmode = AutoSparseForwardDiff(; chunksize = 2)))])
    end

    @compile_workload begin
        for prob in probs, alg in algs
            solve(prob, alg; nlsolve_kwargs = (; abstol = 1e-2))
        end
    end
end

export Shooting, MultipleShooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export BVPM2, BVPSOL, COLNEW # From ODEInterface.jl

export RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7
export LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5
export MIRKJacobianComputationAlgorithm, BVPJacobianAlgorithm

end
