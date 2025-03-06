# Error Control Adaptivity

Adaptivity helps ensure the quality of the our numerical solution, and when our solution exhibits significant estimating errors, adaptivity automatically refine the mesh based on the error distribution, providing a satisfying solution.

When comes to solving ill-conditioned BVP, for example the singular pertubation problem where the small parameters become extremally small leading to the layers phonemona, the error control adaptivity becomes even more critical, because the minor pertubations can lead to large deviation in the solution. In such cases, adaptivity autimatically figure out where to use refined mesh and where to use coarse mesh to achieve the balance of computational efficiency and accuracy.

BoundaryValuDiffEq.jl support error control adaptivity, and the adaptivity is default as on when using adaptive methods:

```julia
sol = solve(prob, MIRK4(), dt = 0.01, adaptive = true)
```

Actually, BoundaryValueDiffEq.jl supports both defect and global error control adaptivity(while the defect control is the default), to specify different error control metods, we simply need to specify the `controller` keyword in `solve`:

```julia
sol = solve(prob, MIRK4(), dt = 0.01, adaptive = true, controller = GlobalErrorControl())
```

## Error control methods

```@docs
DefectControl
GlobalErrorControl
SequentialErrorControl
HybridErrorControl
```

While we can achieve global error control in different ways, we can use different methods to estimate the global error:

```@docs
HOErrorControl
REErrorControl
```
