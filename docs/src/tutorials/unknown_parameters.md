# Estimate Unknown Parameters in BVP

When there are unknown parameters in boundary value problems, we can estimate the unknown parameters by solving the BVP, it is quite useful in practical applications in dynamical optimizations and inverse problems. This approach allows us to incorporate both the governing differential equations and boundary conditions to infer parameters that may not be directly measurable.

Let's walk through this functionality with an intuitive example. In the following tutorial, we use the [Mathieu equation](https://en.wikipedia.org/wiki/Mathieu_wavelet) which is a second-order differential equation:

```math
y''+(\lambda-2q\cos(2x))y=0
```

where $\lambda$ is the unknown parameter we wish to estimate, `q` is a known real-valued parameter, with boundary conditions when `q=5`:

```math
y'(0)=0,\ y'(\pi)=0
```

The second-order BVP can be transformed into a first-order system of BVP:

```math
\begin{cases}
y_1'=y_2\\
y_2'=-(\lambda-2q\cos(2x))y_1
\end{cases}
```

with boundary conditions of

```math
y_2(0)=0,\ y_2(\pi)=0
```

It is worthnoting that in this system, while we have two differetial equations, it isn't enough to estimate the unknown parameters and guarantee a unique numerical solution with only two given boundary conditions. While under the hood, the parameters are estimated simultaneously with the numerical solution, it makes the boundary value problem an underconstrained BVP if the number of constraints are equal to the number of states, which may result in more than one solution. So we should provide additional constraint $y(0)=1$ from the original equation to make sure unique numerical solution and the estimated parameters are we actually wanted.

With BoundaryValueDiffEq.jl, it's easy to solve boundary value problems with unknown parameters, we can just specify `fit_parameters=true` when constructing the BVP and provide the guess of the unknown parameters in `prob.p`, for example, to estimate the unknown parameters in the above BVP system:

```@example unknown
using BoundaryValueDiffEq, Plots
tspan = (0.0, pi)
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -(p[1] - 10 * cos(2 * t)) * u[1]
end
function bca!(res, u, p)
    res[1] = u[2]
    res[2] = u[1] - 1.0
end
function bcb!(res, u, p)
    res[1] = u[2]
end
guess(p, t) = [cos(4t); -4sin(4t)]
bvp = TwoPointBVProblem(f!, (bca!, bcb!), guess, tspan, [15.0],
    bcresid_prototype = (zeros(2), zeros(1)), fit_parameters = true)
sol = solve(bvp, MIRK4(), dt = 0.05)
plot(sol)
```

after solving the boundary value problem, the estimated unknown parameters can be accessed in the solution

```@example unknown
sol.prob.p
```
