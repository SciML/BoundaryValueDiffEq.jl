# [BVP Functions and Jacobian Types](@id bvpfunctions)

The SciML ecosystem provides an extensive interface for declaring extra functions associated with the boundary value probems's data. In traditional libraries, there is usually only few options: the Jacobian and the Jacobian of boundary conditions. However, we allow for a large array of pre-computed functions to speed up the calculations. This is offered via the `BVPFunction` types, which can be passed to the problems.

## Function Type Definitions

```@docs
SciMLBase.BVPFunction
SciMLBase.DynamicalBVPFunction
```
