# [Common Solver Options (Solve Keyword Arguments)](@id solver_options)

## Iteration Controls

  - `abstol::Number`: The absolute tolerance. Defaults to `1e-3`.
  - `reltol::Number`: The relative tolerance. Defaults to `1e-3`.
  - `defect_threshold`: Monitor of the size of defect norm. Defaults to `0.1`.
  - `odesolve_kwargs`: OrdinaryDiffEq.jl solvers kwargs for passing to ODE solving in shooting methods.
  - `nlsolve_kwargs`: NonlinearSolve.jl solvers kwargs for passing to nonlinear solving in collocation methods and shootingn methods.
