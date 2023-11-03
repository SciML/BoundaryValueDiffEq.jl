module BoundaryValueDiffEq

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes, Adapt, BandedMatrices, DiffEqBase, ForwardDiff, LinearAlgebra,
        NonlinearSolve, PreallocationTools, Preferences, RecursiveArrayTools, Reexport,
        SciMLBase, Setfield, SparseArrays, SparseDiffTools

    import ADTypes: AbstractADType
    import ArrayInterface: matrix_colors,
        parameterless_type, undefmatrix, fast_scalar_indexing
    import ConcreteStructs: @concrete
    import DiffEqBase: solve
    import ForwardDiff: pickchunksize
    import RecursiveArrayTools: ArrayPartition, DiffEqArray
    import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val
    import SparseDiffTools: AbstractSparseADType
    import TruncatedStacktraces: @truncate_stacktrace
    import UnPack: @unpack
end

@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

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
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end
    f1(u, p, t) = [u[2], 0]

    function bc1!(residual, u, p, t)
        residual[1] = u[1][1] - 5
        residual[2] = u[end][1]
    end
    bc1(u, p, t) = [u[1][1] - 5, u[end][1]]

    bc1_a!(residual, ua, p) = (residual[1] = ua[1] - 5)
    bc1_b!(residual, ub, p) = (residual[1] = ub[1])

    bc1_a(ua, p) = [ua[1] - 5]
    bc1_b(ub, p) = [ub[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [
        BVProblem(f1!, bc1!, u0, tspan),
        BVProblem(f1, bc1, u0, tspan),
        TwoPointBVProblem(f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype),
        TwoPointBVProblem(f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype),
    ]

    algs = []

    if Preferences.@load_preference("PrecompileMIRK", true)
        append!(algs,
            [MIRK2(; jac_alg), MIRK3(; jac_alg), MIRK4(; jac_alg),
                MIRK5(; jac_alg), MIRK6(; jac_alg)])
    end

    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    @compile_workload begin
        for prob in probs, alg in algs
            solve(prob, alg; dt = 0.2)
        end
    end
end

export Shooting, MultipleShooting
export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6
export MIRKJacobianComputationAlgorithm, BVPJacobianAlgorithm
# From ODEInterface.jl
export BVPM2, BVPSOL

end
