module BoundaryValueDiffEq

using BoundaryValueDiffEqAscher: BoundaryValueDiffEqAscher, Ascher1, Ascher2, Ascher3,
    Ascher4, Ascher5, Ascher6, Ascher7
using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore,
    AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm, BVPVerbosity,
    DEFAULT_VERBOSE, DefectControl, GlobalErrorControl, HOErrorControl,
    HybridErrorControl, NoErrorControl, REErrorControl, SequentialErrorControl,
    integral
using BoundaryValueDiffEqFIRK: BoundaryValueDiffEqFIRK, LobattoIIIa2, LobattoIIIa3,
    LobattoIIIa4, LobattoIIIa5, LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5,
    LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5, RadauIIa1, RadauIIa2,
    RadauIIa3, RadauIIa5, RadauIIa7
using BoundaryValueDiffEqMIRK: BoundaryValueDiffEqMIRK, MIRK2, MIRK3, MIRK4, MIRK5,
    MIRK6, maxsol, minsol
using BoundaryValueDiffEqMIRKN: BoundaryValueDiffEqMIRKN, MIRKN4, MIRKN6
using BoundaryValueDiffEqShooting: BoundaryValueDiffEqShooting, MultipleShooting, Shooting
using DiffEqBase: DiffEqBase, solve
using OrdinaryDiffEqTsit5: Tsit5
using Reexport: @reexport
using SciMLBase: SciMLBase, BVProblem

# The AD selectors and the SciML common interface that `using BoundaryValueDiffEq` is
# documented to supply on its own: pick an AD backend (docs/src/basics/autodiff.md),
# build any of the five BVP forms (docs/src/basics/bvp_problem.md), solve or `init` it,
# `remake` it for a parameter sweep, drive `MultipleShooting` ensembles
# (docs/src/basics/solve.md), and inspect the return code. Every name stays owned and
# documented by ADTypes and SciMLBase; approved via `reexports_allow` in test/qa/qa.jl.
@reexport using ADTypes: AutoEnzyme, AutoFiniteDiff, AutoForwardDiff, AutoMooncake,
    AutoPolyesterForwardDiff, AutoSparse
@reexport using SciMLBase: BVPFunction, BVProblem, DynamicalBVPFunction, EnsembleProblem,
    EnsembleSerial, EnsembleThreads, ODEFunction, ReturnCode, SecondOrderBVProblem,
    TwoPointBVProblem, TwoPointSecondOrderBVProblem, init, remake, solve, solve!,
    successful_retcode

function SciMLBase.__init(prob::BVProblem; kwargs...)
    return SciMLBase.__init(prob, Shooting(Tsit5()); kwargs...)
end

function SciMLBase.__solve(prob::BVProblem; kwargs...)
    return SciMLBase.__solve(prob, Shooting(Tsit5()); kwargs...)
end

include("extension_algs.jl")

export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6

export Shooting, MultipleShooting

export RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7
export LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5

export MIRKN4, MIRKN6

export Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7

export BVPM2, BVPSOL, COLNEW # From ODEInterface.jl

export BVPJacobianAlgorithm

export BVPVerbosity, DEFAULT_VERBOSE

# Error control adaptivity and the cost-functional helper, owned and documented by
# BoundaryValueDiffEqCore (docs/src/basics/error_control.md, docs/src/basics/solve.md).
export DefectControl, GlobalErrorControl, SequentialErrorControl, HybridErrorControl,
    NoErrorControl
export HOErrorControl, REErrorControl
export integral

export maxsol, minsol

end
