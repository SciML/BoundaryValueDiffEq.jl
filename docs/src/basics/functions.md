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

## Types of Initial Guess

We support the following input types for the initial guess. Note that not all algorithms are
able to make use of these. We tend to not throw error but instead display a warning (if
`verbose = true`) and use the most sensible version of the input.

 1. `AbstractArray{<:Number}`: The initial guess is provided only at the start of the
    problem.
 2. `DiffEqArray`: An initial guess on non-uniform mesh. We expect that the mesh starts at
    `t0` and ends at `tf` and is sorted.
 3. `VectorOfArray`: An initial guess on a uniform mesh, where the mesh endpoints are given
    by the `tspan` of the problem.
 4. `AbstractVector{<:AbstractArray}`: An initial guess on a uniform mesh same as
    `VectorOfArray`.
 5. `initial_guess(p, t)`: a function which returns an initial guess for the solution at
    time `t` given the problem parameters `p`.
