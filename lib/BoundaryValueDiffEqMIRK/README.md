# BoundaryValueDiffEqMIRK.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/BoundaryValueDiffEq/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

BoundaryValueDiffEqMIRK.jl is a component of the [BoundaryValueDiffEq.jl](https://github.com/SciML/BoundaryValueDiffEq.jl) monorepo. It provides the Monotonic Implicit Runge-Kutta (MIRK) collocation methods for solving boundary value problems.
While completely independent and usable on its own, users wanting the full BVP solver suite should use [BoundaryValueDiffEq.jl](https://github.com/SciML/BoundaryValueDiffEq.jl).

## Solvers

- `MIRK2`
- `MIRK3`
- `MIRK4`
- `MIRK5`
- `MIRK6`
- `MIRK6I`

For a full description of the solvers and their options, see the [BVP solver documentation](https://docs.sciml.ai/BoundaryValueDiffEq/stable/).
