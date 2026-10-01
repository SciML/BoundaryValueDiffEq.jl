# [Solving Boundary Value Problems on GPUs](@id gpu)

BoundaryValueDiffEq can parallelize collocation equations or multiple-shooting
intervals on a GPU. This tutorial uses the CUDA backend and the APIs in the current
source tree. The examples require a working NVIDIA GPU and are ordinary Julia code
blocks, so building the documentation does not require GPU hardware.

## Choose the execution path

| Solver package | How to select a GPU solve | Mesh or integration control |
|:--|:--|:--|
| [MIRK](@ref mirk) | Supply a `CuArray` initial state | `dt` sets the initial mesh; mesh adaptivity is supported |
| [FIRK](@ref firk) | Supply a `CuArray` initial state; expanded and nested formulations are supported | `dt` sets the initial mesh; adaptivity depends on the method |
| [MIRKN](@ref mirkn) | Supply a `CuArray` initial state for a second-order BVP | Fixed mesh, `adaptive = false` |
| [Ascher](@ref ascher) | Supply a `CuArray` initial state or set `platform = CUDA.CUDABackend()` | Global error control or a fixed mesh |
| [Multiple shooting](@ref shooting) | Set `platform = CUDA.CUDABackend()`, `device_steps`, and a DiffEqGPU kernel ODE algorithm | Fixed steps per shooting interval |

For MIRK, FIRK and MIRKN, the initial state determines whether the nonlinear solve
uses device storage. An initial-guess function returning a `CuArray`, or a vector
of `CuArray` mesh values, also selects this path. States, residuals, Jacobians and
nonlinear iterates use device arrays; host code still handles setup, solver control
and mesh metadata. CUDA Jacobians use sparse CSR storage.

A CPU initial guess with `platform = CUDA.CUDABackend()` instead selects **hybrid
collocation** for these three packages: collocation work runs on the GPU while the
nonlinear solve and solution storage remain on the CPU. Ascher's GPU `platform`
selects device storage even with a CPU initial guess. Multiple shooting uses its
explicit `platform` and `device_steps` settings.

## Install and load the GPU dependencies

Install the solver packages used below and CUDA's sparse direct solver:

```julia
using Pkg
Pkg.add([
    "BoundaryValueDiffEq", "CUDA", "CUDSS", "DiffEqGPU",
])
```

Only the packages for the chosen method are needed. `DiffEqGPU` is needed for GPU
multiple shooting, while the collocation solvers launch their own
KernelAbstractions kernels. Load `CUDSS` for the CUDA sparse direct solves used in
the collocation examples. Loading it also enables the direct-solve extension for
square multiple-shooting problems; multiple shooting uses GMRES without it.

```julia
using BoundaryValueDiffEqMIRK
using CUDA, CUDSS

@assert CUDA.functional()
CUDA.allowscalar(false)
```

Disabling host scalar indexing helps catch accidental transfers. Scalar indexing
*inside* the RHS and boundary functions below is valid: the resident solver calls
these functions inside device kernels. Keep them kernel-compatible: use scalar
arithmetic and loops, avoid host allocations and I/O, and pass data through `p`
instead of capturing CPU arrays. Out-of-place functions must also return values
that can be constructed on the device, such as tuples or static arrays.

Use `Float32` or `Float64` states and consistent types for times, parameters and
tolerances. The examples use `Float64`. Parameters can be numeric arrays, `isbits`
values, or tuples/named tuples of these; the solver prepares their device storage.

## MIRK: a first-order system on the GPU

Solve the harmonic oscillator

```math
y' = v, \qquad v' = -y, \qquad y(0) = 0, \quad y(1) = \sin(1).
```

The exact solution is ``[y(t), v(t)] = [\sin(t), \cos(t)]``. For a two-point
problem, each boundary function receives the state at its own endpoint. Supply
`bcresid_prototype` to specify the size of each boundary residual.

```julia
function oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end

function left!(r, u, p)
    r[1] = u[1]
    return nothing
end

function right!(r, u, p)
    r[1] = u[1] - sin(one(eltype(u)))
    return nothing
end

prob = TwoPointBVProblem(
    oscillator!, (left!, right!), CuArray([0.1, 0.9]), (0.0, 1.0);
    bcresid_prototype = (CUDA.zeros(Float64, 1), CUDA.zeros(Float64, 1)), nlls = Val(false)
)

jac = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))
sol = solve(prob, MIRK4(; jac_alg = jac); dt = 0.05, adaptive = false, abstol = 1.0e-9)
@assert successful_retcode(sol)
@assert isapprox(Array(sol.u[end]), [sin(1.0), cos(1.0)]; atol = 1.0e-5)

# Transfer only the values needed for CPU analysis or plotting.
midpoint = Array(sol(0.5))
values = Array.(sol.u)
```

`dt` is a keyword of `solve`, while `jac_alg` and `platform` belong to the algorithm
constructor. With `adaptive = false`, check discretization accuracy by decreasing
`dt`. To enable mesh refinement, use, for example:

```julia
adaptive_sol = solve(prob, MIRK4(; jac_alg = jac); dt = 0.1, abstol = 1.0e-7)
@assert successful_retcode(adaptive_sol)
```

For a general `BVProblem`, the boundary function receives a solution-like object
and can use interpolation, for example `u(t[1])[1]` and `u(t[end])[1]`. This differs
from the endpoint-state interface of `TwoPointBVProblem`.

## FIRK: expanded or nested implicit stages

Reuse `prob` and `jac` from the MIRK example:

```julia
using BoundaryValueDiffEqFIRK

expanded_sol = solve(
    prob, RadauIIa3(; jac_alg = jac);
    dt = 0.1, adaptive = false, abstol = 1.0e-9
)
nested_sol = solve(
    prob,
    RadauIIa3(;
        jac_alg = jac, nested_nlsolve = true,
        nested_nlsolve_kwargs = (; abstol = 1.0e-11, reltol = 1.0e-11, maxiters = 30)
    );
    dt = 0.1, adaptive = false, abstol = 1.0e-9
)
@assert successful_retcode(expanded_sol) && successful_retcode(nested_sol)
```

The default expanded formulation includes stage variables in the global nonlinear
system. The nested formulation solves the stages locally with batched Newton
iterations and differentiates the converged stage equations implicitly. On the
resident path, `nested_nlsolve_kwargs` accepts only `abstol`, `reltol` and `maxiters`.
`RadauIIa1`, `LobattoIIIb2` and `LobattoIIIc2` require `adaptive = false` (or
`NoErrorControl()`). Other FIRK methods support mesh adaptivity. MIRK and FIRK
problems with algebraic variables require a fixed mesh; use Ascher for adaptive
BVDAEs.

## MIRKN: solve the second-order equation directly

MIRKN accepts ``y'' = f(y', y, p, t)`` without rewriting it as a first-order system.
Here the state has one component and the two boundary conditions constrain its
position. This example can be run after loading CUDA and CUDSS above.

```julia
using BoundaryValueDiffEqMIRKN

function acceleration!(ddu, du, u, p, t)
    ddu[1] = -u[1]
    return nothing
end

function second_left!(r, du, u, p)
    r[1] = u[1]
    return nothing
end

function second_right!(r, du, u, p)
    r[1] = u[1] - sin(one(eltype(u)))
    return nothing
end

second_prob = TwoPointSecondOrderBVProblem(
    acceleration!, (second_left!, second_right!), CuArray([0.5]), (0.0, 1.0);
    bcresid_prototype = (CUDA.zeros(Float64, 1), CUDA.zeros(Float64, 1)), nlls = Val(false)
)
second_sol = solve(second_prob, MIRKN4(); dt = 0.05, adaptive = false, abstol = 1.0e-9)
@assert successful_retcode(second_sol)

# MIRKN solution values contain position first, then velocity.
position = Array(second_sol.u[end].x[1])
velocity = Array(second_sol.u[end].x[2])
@assert isapprox(position, [sin(1.0)]; atol = 1.0e-5)
@assert isapprox(velocity, [cos(1.0)]; atol = 1.0e-5)
```

MIRKN uses a fixed mesh and `NoErrorControl()`. Refine `dt` to check accuracy;
decreasing the nonlinear tolerance alone does not refine the mesh. Interpolation
between its mesh values is linear.

## Ascher: endpoint and multipoint conditions

The two-point oscillator `prob` can also be solved with Ascher:

```julia
using BoundaryValueDiffEqAscher

ascher_sol = solve(prob, Ascher3(); dt = 0.1, adaptive = false, abstol = 1.0e-9)
@assert successful_retcode(ascher_sol)
```

For a standard Ascher `BVProblem`, provide one `zeta` location per boundary
condition. Its boundary callback receives a **local state**, not an interpolating
solution object: residual component `i` is evaluated using the state at `zeta[i]`.
The following imposes the same two endpoint conditions:

```julia
function side_conditions!(r, u, p, t)
    r[1] = u[1]
    r[2] = u[1] - sin(one(t))
    return nothing
end

side_prob = BVProblem(oscillator!, side_conditions!, [0.1, 0.9], (0.0, 1.0))
side_sol = solve(
    side_prob, Ascher3(; platform = CUDA.CUDABackend(), zeta = [0.0, 1.0]);
    dt = 0.1, abstol = 1.0e-7
)
@assert successful_retcode(side_sol)
```

Here the CPU initial guess is moved to the selected GPU backend. Ascher supports
`GlobalErrorControl()` (the default) or `NoErrorControl()` with `adaptive = false`.
Its device path requires vector states and a square, unconstrained nonlinear BVP;
parameter tuning and optimization are unsupported. For BVDAEs, the number of
boundary conditions must equal the number of differential variables.

## Multiple shooting with DiffEqGPU

GPU multiple shooting integrates the intervals in parallel with a DiffEqGPU
kernel algorithm. Reuse `oscillator!`, `left!` and `right!` above:

```julia
using BoundaryValueDiffEqShooting
using DiffEqGPU

shooting_prob = TwoPointBVProblem(
    oscillator!, (left!, right!), [0.1, 0.9], (0.0, 1.0);
    bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
)
shooting_alg = MultipleShooting(
    16, DiffEqGPU.GPUTsit5(); platform = CUDA.CUDABackend(), device_steps = 8
)
shooting_sol = solve(shooting_prob, shooting_alg; abstol = 1.0e-9)
@assert successful_retcode(shooting_sol)
@assert isapprox(Array(shooting_sol.u[end]), [sin(1.0), cos(1.0)]; atol = 1.0e-6)
```

`16` is the number of intervals and `device_steps = 8` specifies eight fixed ODE
steps per interval. `abstol` controls the nonlinear solve, so verify integration
accuracy by increasing `device_steps` or `nshoots`. The returned solution contains
shooting-node values and uses cubic Hermite interpolation between them.

This path requires vector `Float32`/`Float64` states. Both in-place and out-of-place
RHS functions are supported; interval states and parameters are converted to
static storage for DiffEqGPU. `grid_coarsening` defaults to `false` when
`device_steps` is set and must remain false. There is no final single-shooting
solve. Do not pass `odesolve_kwargs`, callbacks, optimization constraints,
`tune_parameters`, or a nonidentity mass matrix on this path. Use `nlsolve_kwargs`
for additional nonlinear solver options. An OrdinaryDiffEq `Tsit5()` is accepted
for the fixed-step **CPU** path, but GPU execution requires a DiffEqGPU algorithm
such as `GPUTsit5()` or `GPUVern7()`.

## Jacobians, linear solvers and practical limits

The device paths support `AutoForwardDiff()` and `AutoFiniteDiff()`, optionally
wrapped in `AutoSparse`. Finite differences support forward and central schemes:

```julia
fd_jac = BVPJacobianAlgorithm(AutoFiniteDiff(; fdjtype = Val(:central)))
fd_sol = solve(prob, MIRK4(; jac_alg = fd_jac); dt = 0.05, adaptive = false)
```

For two-point problems, `BVPJacobianAlgorithm.diffmode` controls differentiation.
For general problems, use `bc_diffmode` and `nonbc_diffmode` to configure boundary
and interior residuals separately. See [Automatic Differentiation Backends](@ref).

Leave `nlsolve = nothing` to use the package's device-aware default. A custom
nonlinear solver and its linear solver must support the device arrays and sparse
matrix format. Resident least-squares problems use LSMR by default; supply an
appropriately sized boundary residual prototype and `nlls = Val(true)`. Ascher's
device path supports square systems only. Multiple shooting also accepts
`device_linsolve` for its default Newton solver on square systems; it cannot be
combined with an explicit `nlsolve`.

Resident MIRK, FIRK and MIRKN solves do not accept optimization solvers or
optimization constraints. MIRK and FIRK support parameter tuning with an in-place
RHS, vector states and numeric vector parameters; MIRKN, Ascher and device
multiple shooting do not. For host optimization with GPU collocation, use the
hybrid path described above where supported.

GPU setup and kernel launches can dominate small BVPs, including this oscillator.
Measure representative meshes after a warm-up solve, synchronize with
`CUDA.@sync` when timing, and compare CPU and GPU solutions at the same accuracy.
Copy results to the CPU with `Array` only when needed. The sparse adapters and
examples here target CUDA; availability of a KernelAbstractions backend alone
does not guarantee support for that backend's sparse linear solves.
