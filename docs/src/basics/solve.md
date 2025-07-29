# [Common Solver Options (Solve Keyword Arguments)](@id solver_options)

  - `abstol::Number`: The absolute tolerance. Defaults to `1e-6`.
  - `adaptive::Bool`: Whether the error control adaptivity is on, default as `true`.
  - `controller`: Error controller for collocation methods, default as `DefectControl()`, more controller options in [Error Control Adaptivity](@ref error_control).
  - `defect_threshold`: Monitor of the size of defect norm. Defaults to `0.1`.
  - `odesolve_kwargs`: OrdinaryDiffEq.jl solvers kwargs for passing to ODE solving in shooting methods. For more information, see the documentation for OrdinaryDiffEq: [Common Solver Options](https://docs.sciml.ai/DiffEqDocs/latest/basics/common_solver_opts/).
  - `nlsolve_kwargs`: NonlinearSolve.jl solvers kwargs for passing to nonlinear solving in collocation methods and shooting methods. For more information, see the documentation for NonlinearSolve: [Common Solver Options](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/). The default absolute tolerance of nonlinear solving in collocaio
  - `verbose`:  Toggles whether warnings are thrown when the solver exits early. Defaults to `true`.
  - `ensemblealg`: Whether `MultipleShooting` uses multithreading, default as `EnsembleThreads()`. For more information, see the documentation for OrdinaryDiffEq: [EnsembleAlgorithms](https://docs.sciml.ai/DiffEqDocs/latest/features/ensemble/#EnsembleAlgorithms).
