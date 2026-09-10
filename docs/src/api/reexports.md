# [Reexported API](@id reexports)

Each BoundaryValueDiffEq package exports a small, deliberate slice of the SciML common
interface on top of its own solver algorithms, so that `using BoundaryValueDiffEq` (or
`using` a single solver sublibrary) is enough to write the documented workflow without a
second import line. None of these names are owned or documented here — this page says
what each package forwards and who owns it, so you know where to go for the real
documentation.

The lists below are kept in sync with the reexport `export` block in each package's
`src/<Package>.jl` and with the approved allow-lists in `test/qa/reexports.jl`, which the
QA suite checks on every run.

Owners linked from this page:

  - [SciMLBase](https://docs.sciml.ai/SciMLBase/stable/) — problem, function and solution
    types and the `solve`/`init`/`remake` interface.
  - [ADTypes](https://sciml.github.io/ADTypes.jl/stable/) — the `Auto*` automatic
    differentiation backend selectors.
  - [NonlinearSolveFirstOrder](https://docs.sciml.ai/NonlinearSolve/stable/) — the
    nonlinear solver algorithms accepted by every solver's `nlsolve` keyword.
  - `BoundaryValueDiffEqCore` — the shared BVP configuration types, documented under
    [Automatic Differentiation Backends](@ref),
    [Error Control Adaptivity](@ref error_control),
    [Verbosity Control](@ref) and
    [Common Solver Options (Solve Keyword Arguments)](@ref solver_options).

## BoundaryValueDiffEq

`using BoundaryValueDiffEq` reexports the solver algorithms of every sublibrary
(`MIRK2`–`MIRK6`, `RadauIIa*`, `LobattoIII*`, `MIRKN4`/`MIRKN6`, `Ascher1`–`Ascher7`,
`Shooting`, `MultipleShooting`, plus `maxsol`/`minsol` and the ODEInterface wrappers),
and in addition:

  - AD backend selectors, from ADTypes: `AutoEnzyme`, `AutoFiniteDiff`, `AutoForwardDiff`,
    `AutoMooncake`, `AutoPolyesterForwardDiff`, `AutoSparse`
  - Jacobian and verbosity configuration, from BoundaryValueDiffEqCore:
    `BVPJacobianAlgorithm`, `BVPVerbosity`, `DEFAULT_VERBOSE`
  - Error control adaptivity, from BoundaryValueDiffEqCore: `DefectControl`,
    `GlobalErrorControl`, `SequentialErrorControl`, `HybridErrorControl`, `NoErrorControl`,
    `HOErrorControl`, `REErrorControl`
  - Cost functionals, from BoundaryValueDiffEqCore: `integral`
  - Problems, from SciMLBase: `BVProblem`, `TwoPointBVProblem`, `SecondOrderBVProblem`,
    `TwoPointSecondOrderBVProblem`, `EnsembleProblem`
  - Functions, from SciMLBase: `BVPFunction`, `DynamicalBVPFunction`, `ODEFunction`
  - Solving, from SciMLBase: `solve`, `solve!`, `init`, `remake`
  - Ensemble algorithms for `MultipleShooting`, from SciMLBase: `EnsembleSerial`,
    `EnsembleThreads`
  - Return status, from SciMLBase: `ReturnCode`, `successful_retcode`

## BoundaryValueDiffEqMIRK and BoundaryValueDiffEqFIRK

In addition to `MIRK2`–`MIRK6`, `MIRK6I`, `maxsol`, `minsol` (MIRK) and the `RadauIIa*` /
`LobattoIII*` algorithms (FIRK), both packages reexport:

  - AD backend selectors, from ADTypes: `AutoEnzyme`, `AutoFiniteDiff`, `AutoForwardDiff`,
    `AutoMooncake`, `AutoPolyesterForwardDiff`, `AutoSparse`
  - Configuration, from BoundaryValueDiffEqCore: `BVPJacobianAlgorithm`, `BVPVerbosity`,
    `DEFAULT_VERBOSE`, `integral`
  - Error control adaptivity, from BoundaryValueDiffEqCore: `DefectControl`,
    `GlobalErrorControl`, `SequentialErrorControl`, `HybridErrorControl`, `NoErrorControl`,
    `HOErrorControl`, `REErrorControl`
  - Inner nonlinear solvers for the `nlsolve` keyword, from NonlinearSolveFirstOrder:
    `GaussNewton`, `LevenbergMarquardt`, `NewtonRaphson`, `TrustRegion`
  - Problems and functions, from SciMLBase: `BVProblem`, `TwoPointBVProblem`,
    `EnsembleProblem`, `BVPFunction`, `ODEFunction`
  - Solving, from SciMLBase: `solve`, `solve!`, `init`, `remake`
  - Return status, from SciMLBase: `ReturnCode`, `successful_retcode`

## BoundaryValueDiffEqAscher

In addition to `Ascher1`–`Ascher7`, the same set as MIRK and FIRK, except that Ascher
documents no ensemble workflow and so does not reexport `EnsembleProblem` or
`ODEFunction`:

  - AD backend selectors, from ADTypes: `AutoEnzyme`, `AutoFiniteDiff`, `AutoForwardDiff`,
    `AutoMooncake`, `AutoPolyesterForwardDiff`, `AutoSparse`
  - Configuration, from BoundaryValueDiffEqCore: `BVPJacobianAlgorithm`, `BVPVerbosity`,
    `DEFAULT_VERBOSE`, `integral`
  - Error control adaptivity, from BoundaryValueDiffEqCore: `DefectControl`,
    `GlobalErrorControl`, `SequentialErrorControl`, `HybridErrorControl`, `NoErrorControl`,
    `HOErrorControl`, `REErrorControl`
  - Inner nonlinear solvers, from NonlinearSolveFirstOrder: `GaussNewton`,
    `LevenbergMarquardt`, `NewtonRaphson`, `TrustRegion`
  - Problems and functions, from SciMLBase: `BVProblem`, `TwoPointBVProblem`,
    `BVPFunction` — the last of which carries the mass matrix for the semi-explicit BVDAE
    form Ascher solves
  - Solving, from SciMLBase: `solve`, `solve!`, `init`, `remake`
  - Return status, from SciMLBase: `ReturnCode`, `successful_retcode`

## BoundaryValueDiffEqMIRKN

MIRKN solves second order boundary value problems, so it reexports the second order
problem and function types rather than the first order ones, and — having no defect
control adaptivity — only `NoErrorControl` from the error controllers:

  - AD backend selectors, from ADTypes: `AutoEnzyme`, `AutoFiniteDiff`, `AutoForwardDiff`,
    `AutoMooncake`, `AutoPolyesterForwardDiff`, `AutoSparse`
  - Configuration, from BoundaryValueDiffEqCore: `BVPJacobianAlgorithm`, `BVPVerbosity`,
    `DEFAULT_VERBOSE`, `NoErrorControl`, `integral`
  - Inner nonlinear solvers, from NonlinearSolveFirstOrder: `GaussNewton`,
    `LevenbergMarquardt`, `NewtonRaphson`, `TrustRegion`
  - Problems and functions, from SciMLBase: `SecondOrderBVProblem`,
    `TwoPointSecondOrderBVProblem`, `DynamicalBVPFunction`
  - Solving, from SciMLBase: `solve`, `solve!`, `init`, `remake`
  - Return status, from SciMLBase: `ReturnCode`, `successful_retcode`

## BoundaryValueDiffEqShooting

Shooting reduces the BVP to initial value problems, so it reexports `ODEProblem` and the
ensemble algorithms accepted by `MultipleShooting`'s `ensemblealg` keyword. It is not a
collocation method and takes no `controller`, so the error control types are not part of
its surface:

  - AD backend selectors, from ADTypes: `AutoEnzyme`, `AutoFiniteDiff`, `AutoForwardDiff`,
    `AutoMooncake`, `AutoPolyesterForwardDiff`, `AutoSparse`
  - Configuration, from BoundaryValueDiffEqCore: `BVPJacobianAlgorithm`, `BVPVerbosity`,
    `DEFAULT_VERBOSE`, `integral`
  - Inner nonlinear solvers, from NonlinearSolveFirstOrder: `GaussNewton`,
    `LevenbergMarquardt`, `NewtonRaphson`, `TrustRegion`
  - Problems and functions, from SciMLBase: `BVProblem`, `TwoPointBVProblem`,
    `ODEProblem`, `BVPFunction`
  - Ensemble algorithms, from SciMLBase: `EnsembleSerial`, `EnsembleThreads`
  - Solving, from SciMLBase: `solve`, `solve!`, `init`, `remake`
  - Return status, from SciMLBase: `ReturnCode`, `successful_retcode`

## BoundaryValueDiffEqCore

Core is the shared base the solver sublibraries build on; it owns
`BVPJacobianAlgorithm`, the error control types, `BVPVerbosity`, `DEFAULT_VERBOSE` and
`integral`, and forwards:

  - Nonlinear solvers, from NonlinearSolveFirstOrder: `GaussNewton`, `LevenbergMarquardt`,
    `NewtonRaphson`, `TrustRegion`
  - Problems, from SciMLBase: `BVProblem`, `TwoPointBVProblem`, `SecondOrderBVProblem`,
    `TwoPointSecondOrderBVProblem`, `NonlinearProblem`, `NonlinearLeastSquaresProblem`,
    `OptimizationProblem`
  - Functions, from SciMLBase: `BVPFunction`, `DynamicalBVPFunction`, `NonlinearFunction`,
    `OptimizationFunction`
  - Solving, from SciMLBase: `solve`, `init`, `remake`
  - Return status, from SciMLBase: `ReturnCode`, `successful_retcode`

## Boundary

These lists are the whole of what these packages forward. Anything else from SciMLBase
must be imported from SciMLBase directly, anything else from ADTypes from ADTypes, and
anything else from NonlinearSolve from NonlinearSolve. In particular, solution types
(`ODESolution` and friends), the integrator interface, callbacks and ensemble analysis
are deliberately not reexported: BVP solvers return a solution object you index and
interpolate rather than an integrator you step, so those names belong to their owners.
