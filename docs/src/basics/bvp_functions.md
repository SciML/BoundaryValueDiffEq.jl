# [BVP Functions and Jacobian Types](@id bvpfunctions)

The SciML ecosystem provides an extensive interface for declaring extra functions associated with the boundary value problem's data. In traditional libraries, there is usually only few options: the Jacobian and the Jacobian of boundary conditions. However, we allow for a large array of pre-computed functions to speed up the calculations. This is offered via the `BVPFunction` types, which can be passed to the problems.

## Function Type Definitions

`BVPFunction` and `DynamicalBVPFunction` are defined and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/). Use their owner documentation for
the full problem-function interface.
