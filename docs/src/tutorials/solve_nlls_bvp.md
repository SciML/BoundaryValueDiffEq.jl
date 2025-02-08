# Solve Overdetermined and Underdetermined BVP

When there are more or less boundary conditions than the states in a boundary value problem, the BVP would become an overdetermined or underdetermined boundary value problem. As for these kinds of special BVPs, the solving workflow are similar with solving standard BVPs in BoundaryValueDiffEq.jl, but we need to specify the prototype of boundary conditions to tell BoundaryValueDiffEq.jl the structure of our boundary conditions with `bcresid_prototype` in `BVPFunction`.

## Solve Overdetermined BVP

For example, consider an overdetermined BVP given by the system of differential equations

```math
y_1'=y_2\\
y_2'=-y_1
```

with boundary conditions of

```math
y_1(0)=0, y_1(100)=1, y_2(100) = -1.729109
```

The test BVP has two state variables but three boundary conditions, which means there are additional constraints on the solution.

```@example nlls_overdetermined
using BoundaryValueDiffEq, Plots
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end
function bc!(resid, sol, p, t)
    solₜ₁ = sol(0.0)
    solₜ₂ = sol(100.0)
    resid[1] = solₜ₁[1]
    resid[2] = solₜ₂[1] - 1
    resid[3] = solₜ₂[2] + 1.729109
end
tspan = (0.0, 100.0)
u0 = [0.0, 1.0]
prob = BVProblem(BVPFunction(f!, bc!; bcresid_prototype = zeros(3)), u0, tspan)
sol = solve(prob, MIRK4(), dt = 0.01)
plot(sol)
```

Since this BVP imposes constaints only at the two endpoints, we can use `TwoPointBVProlem` to handle such cases.

```@example nlls_overdetermined
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end
bca!(resid, ua, p) = (resid[1] = ua[1])
bcb!(resid, ub, p) = (resid[1] = ub[1] - 1; resid[2] = ub[2] + 1.729109)
prob = TwoPointBVProblem(
    BVPFunction(
        f!, (bca!, bcb!); twopoint = Val(true), bcresid_prototype = (zeros(1), zeros(2))),
    u0,
    tspan)
```

## Solve Underdetermined BVP

Let's see an example of underdetermined BVP, consider an horizontal metal beam of length $L$ subject to a vertical load $q(x)$ per unit length, the resulting beam displacement satisfies the differential equation

```math
EIy'(x)=q(x)
```

with boundary condition $y(0)=y(L)=0$, $E$ is the Young's modulus and $I$ is the moment of inertia of the beam's cross section. Here we consider the simplified version and transform this BVP into a first order BVP system:

```math
y_1'=y_2\\
y_2'=y_3\\
y_3'=y_4\\
y_4'=0
```

```@example nlls_underdetermined
using BoundaryValueDiffEq, Plots
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[3]
    du[3] = u[4]
    du[4] = 0
end
function bc!(resid, sol, p, t)
    solₜ₁ = sol(0.0)
    solₜ₂ = sol(1.0)
    resid[1] = solₜ₁[1]
    resid[2] = solₜ₂[1]
end
xspan = (0.0, 1.0)
u0 = [0.0, 1.0, 0.0, 1.0]
prob = BVProblem(BVPFunction(f!, bc!; bcresid_prototype = zeros(2)), u0, xspan)
sol = solve(prob, MIRK4(), dt = 0.01)
plot(sol)
```

Since this problem has less constraints than the state variables, so there would be infinitely many solutions with different `u0` specified.

The above underdetermined is also being able to reformulated as `TwoPointBVProblem`

```@example nlls_underdetermined
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[3]
    du[3] = u[4]
    du[4] = 0
end
bca!(resid, ua, p) = (resid[1] = ua[1])
bcb!(resid, ub, p) = (resid[1] = ub[1])
xspan = (0.0, 1.0)
u0 = [0.0, 1.0, 0.0, 1.0]
prob = TwoPointBVProblem(
    BVPFunction(
        f!, (bca!, bcb!); twopoint = Val(true), bcresid_prototype = (zeros(1), zeros(1))),
    u0,
    xspan)
```
