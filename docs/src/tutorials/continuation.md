# Solve BVP with Continuation

Continuation is a commonly used technique for solving numerically difficult boundary value problems, we exploit the priori knowledge of the solution as initial guess to accelerate the BVP solving by breaking up the difficult BVP into a sequence of simpler problems. For example, we use the problem from [ascher1995numerical](@Citet) in this tutorial:

```math
ε y'' + xy' = ε \pi^2\cos(\pi x) - \pi x\sin(\pi x)
```

for ``ε = 10^{-4}``, on ``t\in[-1,1]`` with two point boundary conditions ``y(-1)=-2``, ``y(1)=0``. With analytical solution of ``y(x) = \cos(\pi x) + \operatorname{erf}(x/\sqrt{2ε})/\operatorname{erf}(1/\sqrt{2ε})``, this problem has a rapid transition layer at ``x=0``, making it difficult to solve numerically. In this tutorial, we will showcase how to use continuation with BoundaryValueDiffEq.jl to solve this BVP.

We use the substitution to transform this problem into a first order BVP system:

```math
\begin{align*}
y_1'&= y_2 \\
y_2'&= -\frac{x}{ε} y_2 - \pi^2\cos(\pi x) - \frac{\pi x}{ε} \sin(\pi x)
\end{align*}
```

Since this BVP would become difficult to solve when ``0<ε\ll 1``, we start the continuation with relatively bigger ``ε`` to first obtain a good initial guess for cases when $\epsilon$ are becoming extremely small. We can just use the previous solution from BVP solving as the initial guess `u0` when constructing a new `BVProblem`.

```@example continuation
using BoundaryValueDiffEq, Plots
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -t / p * u[2] - pi^2 * cospi(t) - pi * t / p * sinpi(t)
end
function bc!(res, u, p, t)
    res[1] = u[1][1] + 2
    res[2] = u[end][1]
end
tspan = (-1.0, 1.0)
sol = [1.0, 0.0]
e = 0.1
for i in 2:4
    global e = e / 10
    prob = BVProblem(f!, bc!, sol, tspan, e)
    global sol = solve(prob, MIRK4(), dt = 0.01)
end
plot(sol, idxs = [1])
```

In the iterative solving, the intermediate solutions are each used as the initial guess for the next problem solving.

## On providing initial guess

There are several ways of providing initial guess in `BVProblem`/`TwoPointBVProblem`:

 1. Solution from BVP/ODE solving from SciML packages with `ODESolution` type.
 2. `VectorOfArray` from RecursiveArrayTools.jl.
 3. `DiffEqArray` from RecursiveArrayTools.jl.
 4. Function handle of the form `f(p, t)` for specifying initial guess on time span.
 5. `AbstractArray` represent only the possible initial condition.
