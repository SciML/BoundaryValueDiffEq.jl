# BoundaryValueDiffEq

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DiffEqDocs/stable/)

[![Build Status](https://github.com/SciML/BoundaryValueDiffEq.jl/workflows/CI/badge.svg)](https://github.com/SciML/BoundaryValueDiffEq.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/SciML/BoundaryValueDiffEq.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/BoundaryValueDiffEq.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/BoundaryValueDiffEq)](https://pkgs.genieframework.com?packages=BoundaryValueDiffEq)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

BoundaryValueDiffEq.jl is a component package in the DifferentialEquations ecosystem. It holds the
boundary value problem solvers and utilities. While completely independent
and usable on its own, users interested in using this
functionality should check out [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl).

## API

BoundaryValueDiffEq.jl is part of the SciML common interface, but can be used independently of DifferentialEquations.jl. The only requirement is that the user passes a BoundaryValueDiffEq.jl algorithm to solve. For example, we can solve the [BVP tutorial from the documentation](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/bvp_example/) using the `MIRK4()` algorithm:

```julia
using BoundaryValueDiffEq
tspan = (0.0, pi / 2)
function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -9.81 * sin(θ)
end
function bc!(residual, u, p, t)
    residual[1] = u(pi / 4)[1] + pi / 2
    residual[2] = u(pi / 2)[1] - pi / 2
end
prob = BVProblem(simplependulum!, bc!, [pi / 2, pi / 2], tspan)
sol = solve(prob, MIRK4(), dt = 0.05)
```

## Available Solvers

For the list of available solvers, please refer to the [DifferentialEquations.jl BVP Solvers page](https://docs.sciml.ai/DiffEqDocs/stable/solvers/bvp_solve/). For options for the `solve` command, see the [common solver options page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).

## Controlling Precompilation

Precompilation can be controlled via `Preferences.jl`

  - `PrecompileMIRK` -- Precompile the MIRK2 - MIRK6 algorithms (default: `true`).
  - `PrecompileShooting` -- Precompile the single shooting algorithms (default: `true`).
  - `PrecompileMultipleShooting` -- Precompile the multiple shooting algorithms (default: `true`).
  - `PrecompileMIRKNLLS` -- Precompile the MIRK2 - MIRK6 algorithms for under-determined and over-determined BVPs (default: `false`).
  - `PrecompileShootingNLLS` -- Precompile the single shooting algorithms for under-determined and over-determined BVPs (default: `true`).
  - `PrecompileMultipleShootingNLLS` -- Precompile the multiple shooting algorithms for under-determined and over-determined BVPs (default: `true` ).

To set these preferences before loading the package, do the following (replacing `PrecompileShooting` with the preference you want to set, or pass in multiple pairs to set them together):

```julia
using Preferences, UUIDs
Preferences.set_preferences!(
    UUID("764a87c0-6b3e-53db-9096-fe964310641d"), "PrecompileShooting" => false)
```

## Running Benchmarks Locally

We include a small set of benchmarks in the `benchmarks` folder. These are not extensive and mainly used to track regressions during development. For more extensive benchmarks, see the [SciMLBenchmarks](https://github.com/SciML/SciMLBenchmarks.jl) repository.

To run benchmarks locally install [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) and run the following command in the package directory:

```bash
benchpkg BoundaryValueDiffEq --rev="master,<git sha for your commit>" --bench-on="master"
```
