# [Boundary Value Problems](@id problems)

## The Five Types of Boundary Value Problems

BoundaryValueDiffEq.jl tackles five related types of boundary value problems:

 1. General boundary value problems:, i.e., constraints are applied over the time span. Both overconstraints and underconstraints BVP are supported.
 2. Two-point boundary value problems:, i.e., constraints are only applied at start and end of time span. Both overconstraints and underconstraints BVP are supported.
 3. General second order boundary value problems, i.e., constraints for both solution and derivative of solution are applied over time span. Both overconstraints and underconstraints second order BVP are supported.
 4. Second order two-point boundary value problems, i.e., constraints for both solution and derivative of solution are only applied at the start and end of the time span. Both overconstraints and underconstraints second order BVP are supported.
 5. Boundary value differential algebraic equations, i.e., apart from constraints applied over the time span, BVDAE has additional algebraic equations which state the algebraic relationship of different states in BVDAE.

## Problem Construction Details

```@docs
BVProblem
TwoPointBVProblem
SecondOrderBVProblem
TwoPointSecondOrderBVProblem
```
