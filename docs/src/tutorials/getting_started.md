# Get Started with Efficient BVP solving in Julia

When ordinary differential equations has constraints over the time span, we should model the differential equations as a boundary value problem which has the form of:

```math
\frac{du}{dt}=f(u, p, t)\\
g(u(a),u(b))=0
```

BoundaryValueDiffEq.jl addresses three types of BVProblem.

 1. General boundary value problems:, i.e., differential equations with constraints applied over the time span. This is a system where you would like to obtain the solution of the differential equations and make sure the solution satisfy the boundary conditions simutanously.
 2. General second order boundary value problems, i.e., differential equations with constraints for both solution and derivative of solution applied over time span. This is a system where you would like to obtain the solution of the differential equations and make sure the solution satisfy the boundary conditions simutanously.
 3. Boundary value differential-algebraic equations, i.e., apart from constraints applied over the time span, BVDAE has additional algebraic equations which state the algebraic relationship of different states in BVDAE.

## Solving Linear two-point boundary value problem

Consider the linear two-point boundary value problem from [standard BVP test problem](https://archimede.uniba.it/%7Ebvpsolvers/testsetbvpsolvers/?page_id=29).

```@example getting_started
using BoundaryValueDiffEq
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[1]
end
function bc!(res, u, p, t)
    res[1] = u(0.0)[1] - 1
    res[2] = u(1.0)[1]
end
tspan = (0.0, 1.0)
u0 = [0.0, 0.0]
prob = BVProblem(f!, bc!, u0, tspan)
sol = solve(prob, MIRK4(), dt = 0.01)
```

Since this proble only has constraints at the start and end of the time span, we can directly use `TwoPointBVProblem`:

```@example getting_started
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[1]
end
function bca!(res, ua, p)
    res[1] = ua[1] - 1
end
function bcb!(res, ub, p)
    res[1] = ub[1]
end
tspan = (0.0, 1.0)
u0 = [0.0, 0.0]
prob = TwoPointBVProblem(
    f!, (bca!, bcb!), u0, tspan, bcresid_prototype = (zeros(1), zeros(1)))
sol = solve(prob, MIRK4(), dt = 0.01)
```

## Solving second order boundary value problem

Consirder the test problem from example problems in MIRKN paper [Muir2001MonoImplicitRM](@Citet).

```math
\begin{cases}
y_1'(x)= y_2(x),\\
\epsilon y_2'(x)=-y_1(x)y_2'(x)- y_3(x)y_3'(x),\\
\epsilon y_3'(x)=y_1'(x) y_3(x)- y_1(x) y_3 '(x)
\end{cases}
```

with initial conditions:

```math
y_1(0) = y_1'(0)= y_1(1)=y_1'(1)=0,y_3(0)=
-1, y_3(1)=1
```

```@example getting_started
using BoundaryValueDiffEqMIRKN
function f!(ddu, du, u, p, t)
    ϵ = 0.1
    ddu[1] = u[2]
    ddu[2] = (-u[1] * du[2] - u[3] * du[3]) / ϵ
    ddu[3] = (du[1] * u[3] - u[1] * du[3]) / ϵ
end
function bc!(res, du, u, p, t)
    res[1] = u(0.0)[1]
    res[2] = u(1.0)[1]
    res[3] = u(0.0)[3] + 1
    res[4] = u(1.0)[3] - 1
    res[5] = du(0.0)[1]
    res[6] = du(1.0)[1]
end
u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 1.0)
prob = SecondOrderBVProblem(f!, bc!, u0, tspan)
sol = solve(prob, MIRKN4(), dt = 0.01)
```

## Solving semi-expicit boundary value differential-algebraic equations

Consider the nonlinear semi-explicit DAE of index at most 2 in COLDAE paper [ascher1994collocation](@Citet)

```math
\begin{cases}
x_1'=(\epsilon+x_2-p_2(t))y+p_1'(t) \\
x_2'=p_2'(t) \\
x_3'=y \\
0=(x_1-p_1(t))(y-e^t)
\end{cases}
```

with boundary conditions

```math
x_1(0)=0,x_3(0)=1,x_2(1)=\sin(1)
```

```@example getting_started
using BoundaryValueDiffEqAscher
function f!(du, u, p, t)
    e = 2.7
    du[1] = (1 + u[2] - sin(t)) * u[4] + cos(t)
    du[2] = cos(t)
    du[3] = u[4]
    du[4] = (u[1] - sin(t)) * (u[4] - e^t)
end
function bc!(res, u, p, t)
    res[1] = u[1]
    res[2] = u[3] - 1
    res[3] = u[2] - sin(1.0)
end
u0 = [0.0, 0.0, 0.0, 0.0]
tspan = (0.0, 1.0)
fun = BVPFunction(f!, bc!, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
prob = BVProblem(fun, u0, tspan)
sol = solve(prob, Ascher4(zeta = [0.0, 0.0, 1.0]), dt = 0.01)
```
