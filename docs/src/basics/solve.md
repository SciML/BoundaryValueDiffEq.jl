# [Common Solver Options (Solve Keyword Arguments)](@id solver_options)

## Iteration Controls

  - `maxiters::Int`: The maximum number of iterations to perform. Defaults to `1000`.
  - `maxtime`: The maximum time for solving the nonlinear system of equations. Defaults to
    `nothing` which means no time limit. Note that setting a time limit does have a small overhead.
  - `abstol::Number`: The absolute tolerance. Defaults to `1e-3`.
  - `reltol::Number`: The relative tolerance. Defaults to `1e-3`.
  - `defect_threshold`: Monitor of the size of defect norm. Defaults to `0.1`.
