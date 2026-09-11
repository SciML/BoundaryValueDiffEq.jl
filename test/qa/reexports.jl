# Audited intentional facade reexports: the names each package deliberately makes
# available from `using <Package>` even though another package owns and documents them.
#
# Each list is kept in sync with the reexport `export` block in that package's
# `src/<Package>.jl` and is handed to `run_qa` as `reexports_allow`, so the strict
# "no unapproved public reexports" check cannot drift from the source. Adding a name
# here is an API review; removing one is a breaking change for downstream users who
# get that name from `using <Package>`.

# ADTypes selectors documented in docs/src/basics/autodiff.md.
const AD_REEXPORTS = Symbol[
    :AutoEnzyme,
    :AutoFiniteDiff,
    :AutoForwardDiff,
    :AutoMooncake,
    :AutoPolyesterForwardDiff,
    :AutoSparse,
]

# BoundaryValueDiffEqCore configuration documented in docs/src/basics/autodiff.md and
# docs/src/basics/verbosity.md.
const CORE_CONFIG_REEXPORTS = Symbol[
    :BVPJacobianAlgorithm,
    :BVPVerbosity,
    :DEFAULT_VERBOSE,
]

# Error control adaptivity documented in docs/src/basics/error_control.md, plus the
# cost-functional helper documented in docs/src/basics/solve.md.
const ERROR_CONTROL_REEXPORTS = Symbol[
    :DefectControl,
    :GlobalErrorControl,
    :HOErrorControl,
    :HybridErrorControl,
    :NoErrorControl,
    :REErrorControl,
    :SequentialErrorControl,
]

# NonlinearSolveFirstOrder algorithms accepted by every solver's `nlsolve` keyword.
const NLSOLVE_REEXPORTS = Symbol[
    :GaussNewton,
    :LevenbergMarquardt,
    :NewtonRaphson,
    :TrustRegion,
]

const ROOT_REEXPORTS = Symbol[
    # Solver algorithms owned by the sublibraries.
    :Ascher1, :Ascher2, :Ascher3, :Ascher4, :Ascher5, :Ascher6, :Ascher7,
    :LobattoIIIa2, :LobattoIIIa3, :LobattoIIIa4, :LobattoIIIa5,
    :LobattoIIIb2, :LobattoIIIb3, :LobattoIIIb4, :LobattoIIIb5,
    :LobattoIIIc2, :LobattoIIIc3, :LobattoIIIc4, :LobattoIIIc5,
    :MIRK2, :MIRK3, :MIRK4, :MIRK5, :MIRK6,
    :MIRKN4, :MIRKN6,
    :MultipleShooting, :Shooting,
    :RadauIIa1, :RadauIIa2, :RadauIIa3, :RadauIIa5, :RadauIIa7,
    :maxsol, :minsol,
    # ADTypes selectors and BoundaryValueDiffEqCore configuration.
    AD_REEXPORTS...,
    CORE_CONFIG_REEXPORTS...,
    ERROR_CONTROL_REEXPORTS...,
    :integral,
    # SciML common interface for boundary value problems.
    :BVPFunction, :BVProblem, :DynamicalBVPFunction, :EnsembleProblem, :EnsembleSerial,
    :EnsembleThreads, :ODEFunction, :ReturnCode, :SecondOrderBVProblem, :TwoPointBVProblem,
    :TwoPointSecondOrderBVProblem, :init, :remake, :solve, :solve!, :successful_retcode,
]

const CORE_REEXPORTS = Symbol[
    NLSOLVE_REEXPORTS...,
    :BVPFunction,
    :BVProblem,
    :DynamicalBVPFunction,
    :NonlinearFunction,
    :NonlinearLeastSquaresProblem,
    :NonlinearProblem,
    :OptimizationFunction,
    :OptimizationProblem,
    :ReturnCode,
    :SecondOrderBVProblem,
    :TwoPointBVProblem,
    :TwoPointSecondOrderBVProblem,
    :init,
    :remake,
    :solve,
    :successful_retcode,
]

# MIRK, FIRK and Ascher share the first-order BVP facade: the problem and function
# types they solve, the solve/init interface, error control, AD selectors and the
# inner nonlinear solvers.
const COLLOCATION_REEXPORTS = Symbol[
    AD_REEXPORTS...,
    CORE_CONFIG_REEXPORTS...,
    ERROR_CONTROL_REEXPORTS...,
    NLSOLVE_REEXPORTS...,
    :integral,
    :BVPFunction,
    :BVProblem,
    :ReturnCode,
    :TwoPointBVProblem,
    :init,
    :remake,
    :solve,
    :solve!,
    :successful_retcode,
]

# MIRK and FIRK additionally document ensembles and `ODEFunction`-wrapped dynamics.
const MIRK_REEXPORTS = Symbol[COLLOCATION_REEXPORTS..., :EnsembleProblem, :ODEFunction]
const FIRK_REEXPORTS = MIRK_REEXPORTS

const ASCHER_REEXPORTS = COLLOCATION_REEXPORTS

# MIRKN solves second order BVPs; it has no defect control adaptivity
# (docs/src/solvers/mirkn.md), so only `NoErrorControl` applies.
const MIRKN_REEXPORTS = Symbol[
    AD_REEXPORTS...,
    CORE_CONFIG_REEXPORTS...,
    NLSOLVE_REEXPORTS...,
    :NoErrorControl,
    :integral,
    :DynamicalBVPFunction,
    :ReturnCode,
    :SecondOrderBVProblem,
    :TwoPointSecondOrderBVProblem,
    :init,
    :remake,
    :solve,
    :solve!,
    :successful_retcode,
]

# Shooting reduces the BVP to IVPs, so it exposes `ODEProblem` and the ensemble
# algorithms accepted by `MultipleShooting`'s `ensemblealg` keyword
# (docs/src/basics/solve.md). It is not a collocation method and takes no `controller`.
const SHOOTING_REEXPORTS = Symbol[
    AD_REEXPORTS...,
    CORE_CONFIG_REEXPORTS...,
    NLSOLVE_REEXPORTS...,
    :integral,
    :BVPFunction,
    :BVProblem,
    :EnsembleSerial,
    :EnsembleThreads,
    :ODEProblem,
    :ReturnCode,
    :TwoPointBVProblem,
    :init,
    :remake,
    :solve,
    :solve!,
    :successful_retcode,
]

"""
    test_reexport_surface(pkg, reexports, scope)

Assert that every approved reexport is actually reachable from `using pkg`, so an
allow-list above cannot drift into approving names the package no longer provides.
`scope` is the module whose own `using pkg` has to bring the name into scope, which
tests the property downstream users depend on directly.
"""
function test_reexport_surface(pkg::Module, reexports, scope::Module)
    @testset "Reexport surface" begin
        @testset "$name" for name in reexports
            @test name in names(pkg)
            @test isdefined(scope, name)
        end
    end
    return nothing
end
