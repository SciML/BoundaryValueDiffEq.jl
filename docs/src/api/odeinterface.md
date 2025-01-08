# ODEInterface.jl

This is an extension for importing solvers from
[ODEInterface.jl](https://github.com/luchr/ODEInterface.jl) into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install
the package before using these solvers:

```julia
using Pkg
Pkg.add("ODEInterface")
using ODEInterface, BoundaryValueDiffEq
```

## Solver API

```@docs
BVPM2
BVPSOL
COLNEW
```
