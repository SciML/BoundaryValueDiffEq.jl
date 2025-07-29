# [Error Control Adaptivity](@id error_control)

Adaptivity helps ensure the quality of the our numerical solution, and when our solution exhibits significant estimating errors, adaptivity automatically refine the mesh based on the error distribution, and providing a final satisfying solution.

When comes to solving ill-conditioned BVP, for example the singular perturbation problem where the small parameters become extremely small leading to the layers phonemona, the error control adaptivity becomes even more critical, because the minor perturbations can lead to large deviation in the solution. In such cases, adaptivity automatically figure out where to use refined mesh and where to use coarse mesh to achieve the balance of computational efficiency and accuracy.

BoundaryValuDiffEq.jl support error control adaptivity for collocation methods, and the adaptivity is default as defect control adaptivity when using adaptive collocation solvers:

```julia
sol = solve(prob, MIRK4(), dt = 0.01, adaptive = true)
```

Actually, BoundaryValueDiffEq.jl supports both defect and global error control adaptivity(while the defect control is the default controller) [boisvert2013runge](@Citet), to specify different error control methods, we simply need to specify the `controller` keyword in `solve`:

```julia
sol = solve(prob, MIRK4(), dt = 0.01, controller = GlobalErrorControl()) # Use global error control
sol = solve(prob, MIRK4(), dt = 0.01, controller = SequentialErrorControl()) # Use Sequential error control
sol = solve(prob, MIRK4(), dt = 0.01, controller = HybridErrorControl()) # Use Hybrid error control
```

## Error control methods

```@docs
DefectControl
GlobalErrorControl
SequentialErrorControl
HybridErrorControl
NoErrorControl
```

While we can achieve global error control in different ways, we can use different methods to estimate the global error:

```@docs
HOErrorControl
REErrorControl
```
