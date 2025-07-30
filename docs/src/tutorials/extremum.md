# Solve BVP with Extremum Boundary Conditions

In many physical systems, boundary conditions are not always defined at fixed points such as initial or terminal ends of the domain. Instead, we may encounter scenarios where constraints are imposed on the maximum or minimum values that the solution must attain somewhere within the domain. In such cases, we can use the `maxsol` and `minsol` functions provided by BoundaryValueDiffEq.jl to specify such extremum boundary conditions.

Let's walk through this functionality with an intuitive example. We still revisit the simple pendulum example here, but this time, suppose we need to impose the maximum and minimum value to our boundary conditions, specified as:

```math
\max{u}=ub\\
\min{u}=lb
```

where `lb=-4.8161991710010925` and `ub=5.0496477654230745`. So the states must conform that the maximum value of the state should be `lb` while the minimum value of the state should be `ub`. To solve such problems, we can simply use the `maxsol` and `minsol` functions when defining the boundary value problem in BoundaryValueDiffEq.jl.

```@example inequality
using BoundaryValueDiffEq, Plots
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
