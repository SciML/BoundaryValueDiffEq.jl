# [BVPFunctions and Jacobian Types](@id bvpfunctions)

The SciML ecosystem provides an extensive interface for declaring extra functions
associated with the differential equation's data. In traditional libraries, there is usually
only one option: the Jacobian. However, we allow for a large array of pre-computed functions
to speed up the calculations. This is offered via the `BVPFunction` types, which can
be passed to the problems.

## Function Type Definitions

```@docs
SciMLBase.BVPFunction
```

There is also a convenience function `SciMLBase.TwoPointBVFunction`. Stay tuned for its
docs.
