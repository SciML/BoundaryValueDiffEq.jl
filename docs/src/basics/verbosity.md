# Verbosity Control

## Verbosity Specification with SciMLLogging.jl

BoundaryValueDiffEq.jl uses SciMLLogging.jl to provide users with fine-grained control over logging and diagnostic output during BVP solving. The `BVPVerbosity` struct allows you to customize which messages are displayed, from deprecation warnings to detailed debugging information about solver convergence, linear algebra issues, and internal NonlinearSolve.jl/Optimization.jl solver diagnostics.

## Basic Usage

Pass a `BVPVerbosity` object to `solve` or `init` using the `verbose` keyword argument:

```julia
using BoundaryValueDiffEq

# Define a boundary value problem
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end

function bc!(res, u, p, t)
    res[1] = u[1][1]
    res[2] = u[end][1] - 1
end

u0 = [0.0, 1.0]
tspan = (0.0, 1.0)
prob = BVProblem(f!, bc!, u0, tspan)

# Solve with detailed verbosity to see convergence info
verbose = BVPVerbosity(Detailed())
sol = solve(prob, MIRK4(), dt = 0.1, verbose = verbose)

# Solve with completely silent output (no warnings or deprecations)
sol = solve(prob, MIRK4(), dt = 0.1, verbose = BVPVerbosity(None()))

# Solve with default verbosity (standard preset)
sol = solve(prob, MIRK4(), dt = 0.1)  # equivalent to verbose = BVPVerbosity()
```

## Controlling Internal Solver Verbosity

BoundaryValueDiffEq.jl solvers internally use NonlinearSolve.jl (for nonlinear systems) or Optimization.jl (when using optimization-based methods). You can control the verbosity of these internal solvers independently:

```julia
# Silence BVP messages but show all NonlinearSolve convergence info
verbose = BVPVerbosity(
    None(),
    nonlinear_verbosity = All()
)
sol = solve(prob, MIRK4(), dt = 0.1, verbose = verbose)

# Show standard BVP messages but silence NonlinearSolve output
verbose = BVPVerbosity(
    Standard(),
    nonlinear_verbosity = None()
)
sol = solve(prob, MIRK4(), dt = 0.1, verbose = verbose)

# Control Optimization.jl verbosity when using optimization-based methods
using Optimization, OptimizationOptimJL

verbose = BVPVerbosity(
    Standard(),
    optimization_verbosity = Detailed()
)
sol = solve(prob, MIRK4(optimize = OptimizationOptimJL.BFGS()), dt = 0.1, verbose = verbose)
```

## API Reference

```@docs
BVPVerbosity
```
