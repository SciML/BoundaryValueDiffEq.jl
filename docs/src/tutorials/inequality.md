# Solve BVP with Inequality Boundary Conditions

When dealing with scenarios that boundary conditions are stronger than just explicit values at specific points—such as inequality boundary conditions, where the solution must satisfy constraints like staying within upper and lower bounds—we need a more flexible approach. In such cases, we can use the `maxsol` and `minsol` functions provided by BoundaryValueDiffEq.jl to specify such inequality conditions.

Let's walk through this functionlity with an intuitive example. We still revisit the simple pendulum example here, but this time, we’ll impose upper and lower bound constraints on the solution, specified as:

```math
lb \leq u \leq ub
```

where `lb=-4.8161991710010925` and `ub=5.0496477654230745`. So the states must be bigger than `lb` but smaller than `ub`. To solve such problems, we can simply use the `minsol` and `maxsol` functions when defining the boundary value problem in BoundaryValueDiffEq.jl.

```@example inequality
tspan = (0.0, pi / 2)
function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -9.81 * sin(θ)
end
function bc!(residual, u, p, t)
    residual[1] = maxsol(u, (0.0, pi / 2)) - 5.0496477654230745
    residual[2] = minsol(u, (0.0, pi / 2)) + 4.8161991710010925
end
prob = BVProblem(simplependulum!, bc!, [pi / 2, pi / 2], tspan)
sol = solve(prob, MIRK4(), dt = 0.05)
plot(sol)
```
