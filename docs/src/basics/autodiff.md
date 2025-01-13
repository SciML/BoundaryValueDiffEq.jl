# Automatic Differentiation Backends

!!! note
    
    We support all backends supported by DifferentiationInterface.jl. Please refer to
    the [backends page](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/explanation/backends/)
    for more information.

In BoundaryValueDiffEq.jl, automatic differentiation backend should only be wrapped in `BVPJacobianAlgorithm(diffmode, bc_diffmode, nonbc_diffmode)`. `BVPJacobianAlgorithm(diffmode, bc_diffmode, nonbc_diffmode)` supports user-specified mixed automatic differentiation backends in differrent part of a boundary value problem, and AD choice should depended on the type of boundary value problem:

  - [`BVProblem`](@ref SciMLBase.BVProblem): Differentiation mode for boundary condition part and non boundary condition part should be specified, for example, `BVPJacobianAlgorithm(; bc_diffmode, nonbc_diffmode)`, default to `BVPJacobianAlgorithm(; bc_diffmode = AutoForwardDiff(), nonbc_diffmode = AutoSparse(AutoForwardDiff()))`.
  - [`TwoPointBVProblem`](@ref SciMLBase.TwoPointBVProblem): Differentiation mode for overall solving should be specified, for example, `BVPJacobianAlgorithm(; diffmode)`, default to `BVPJacobianAlgorithm(; diffmode = AutoSparse(AutoForwardDiff()))`.

In BoundaryValueDiffEq.jl, we require AD to obtain the Jacobian of the loss function which contains the collocation equation and boundary condition equations. For `TwoPointBVProblem`, the Jacobian of the loss function is a sparse banded matrix with known sparsity pattern, but for general multi-points `BVProblem`, the Jacobian of the loss function is an almost banded matrix, which has the first several rows as the boundary conditions and the rest as a sparse banded matrix with known sparsity pattern but without the first several rows. In this case, we can specify mixed AD backend in `BVPJacobianAlgorithm` to make the most of the different sparsity pattern to accelerate BVP solving process.

## Summary of Finite Differencing Backends

  - [`AutoFiniteDiff`](@extref ADTypes.AutoFiniteDiff): Finite differencing using
    `FiniteDiff.jl`, not optimal but always applicable.
  - [`AutoFiniteDifferences`](@extref ADTypes.AutoFiniteDifferences): Finite differencing
    using `FiniteDifferences.jl`, not optimal but always applicable.

## Summary of Forward Mode AD Backends

  - [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff): The best choice for dense
    problems.
  - [`AutoPolyesterForwardDiff`](@extref ADTypes.AutoPolyesterForwardDiff): Might be faster
    than [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff) for large problems. Requires
    `PolyesterForwardDiff.jl` to be installed and loaded.

## Summary of Reverse Mode AD Backends

  - [`AutoZygote`](@extref ADTypes.AutoZygote): The fastest choice for non-mutating
    array-based (BLAS) functions.
  - [`AutoEnzyme`](@extref ADTypes.AutoEnzyme): Uses `Enzyme.jl` Reverse Mode and works for
    both in-place and out-of-place functions.
