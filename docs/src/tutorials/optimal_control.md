# Solve Optimal Control problem

A classical optimal control problem is the rocket launching problem(aka [Goddard Rocket problem](https://en.wikipedia.org/wiki/Goddard_problem)). Say we have a rocket with limited fuel and is launched vertically. And we want to control the final altitude of this rocket so that we can make the best of the limited fuel in rocket to get to the highest altitude. In this optimal control problem, the state variables are:

  - Velocity of the rocket: $x_v(t)$
  - Altitude of the rocket: $x_h(t)$
  - Mass of the rocket and the fuel: $x_m(t)$

The control variable is

  - Thrust of the rocket: $u_t(t)$

The dynamics of the launching can be formulated with three differential equations:

$$
\left\{\begin{aligned}
&\frac{dx_v}{dt}=\frac{u_t-drag(x_h,x_v)}{x_m}-g(x_h)\\
&\frac{dx_h}{dt}=x_v\\
&\frac{dx_m}{dt}=-\frac{u_t}{c}
\end{aligned}\right.
$$

where the drag $D(x_h,x_v)$ is a function of altitude and velocity:

$$
D(x_h,x_v)=D_c\cdot x_v^2\cdot\exp^{h_c(\frac{x_h-x_h(0)}{x_h(0)})}
$$

gravity $g(x_h)$ is a function of altitude:

$$
g(x_h)=g_0\cdot (\frac{x_h(0)}{x_h})^2
$$

$c$ is a constant. Suppose the final time is $T$, we here want to maximize the final altitude $x_h(T)$:

$$
\max x_h(T)
$$

The inequality constraints for the state variables and control variables are:

$$
\left\{\begin{aligned}
&x_v>0\\
&x_h>0\\
&m_T<x_m<m_0\\
&0<u_t<u_{t\text{max}}
\end{aligned}\right.
$$

Similar solving for such optimal control problem can be found on JuMP.jl and InfiniteOpt.jl. The detailed parameters are taken from [COPS](https://www.mcs.anl.gov/%7Emore/cops/cops3.pdf).

```julia
using BoundaryValueDiffEqMIRK, OptimizationIpopt, Plots
h_0 = 1                      # Initial height
v_0 = 0                      # Initial velocity
m_0 = 1.0                    # Initial mass
m_T = 0.6                    # Final mass
g_0 = 1                      # Gravity at the surface
h_c = 500                    # Used for drag
c = 0.5 * sqrt(g_0 * h_0)    # Thrust-to-fuel mass
D_c = 0.5 * 620 * m_0 / g_0  # Drag scaling
u_t_max = 3.5 * g_0 * m_0    # Maximum thrust
T_max = 0.2                  # Number of seconds
T = 1_000                    # Number of time steps
Δt = 0.2 / T;                # Time per discretized step

tspan = (0.0, 0.2)
drag(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
g(x_h) = g_0 * (h_0 / x_h)^2
function rocket_launch!(du, u, p, t)
    # u_t is the control variable (thrust)
    x_v, x_h, x_m, u_t = u[1], u[2], u[3], u[4]
    du[1] = (u_t-drag(x_h, x_v))/x_m - g(x_h)
    du[2] = x_v
    du[3] = -u_t/c
end
function rocket_launch_bc!(res, u, p, t)
    res[1] = u(0.0)[1] - v_0
    res[2] = u(0.0)[2] - h_0
    res[3] = u(0.0)[3] - m_0
    res[4] = u(0.2)[4] - 0.0
end
cost_fun(u, p) = -u[end - 2] #Final altitude x_h. To minimize, only temporary, need to use temporary solution interpolation here similar to what we do in boundary condition evaluations.
u0 = [v_0, h_0, m_T, 3.0]
rocket_launch_fun = BVPFunction(rocket_launch!, rocket_launch_bc!; cost = cost_fun, f_prototype = zeros(3))
rocket_launch_prob = BVProblem(
    rocket_launch_fun, u0, tspan; lb = [0.0, h_0, m_T, 0.0], ub = [Inf, Inf, m_0, u_t_max])
sol = solve(rocket_launch_prob, MIRK4(; optimize = Ipopt.Optimizer()); dt = Δt, adaptive = false)

u = reduce(hcat, sol.u)
v, h, m, c = u[1, :], u[2, :], u[3, :], u[4, :]

# Plot the solution
p1 = plot(sol.t, v, xlabel = "Time", ylabel = "Velocity", legend = false)
p2 = plot(sol.t, h, xlabel = "Time", ylabel = "Altitude", legend = false)
p3 = plot(sol.t, m, xlabel = "Time", ylabel = "Mass", legend = false)
p4 = plot(sol.t, c, xlabel = "Time", ylabel = "Thrust", legend = false)

plot(p1, p2, p3, p4, layout = (2, 2))
```

Similar optimal control problem solving can also be deployed in JuMP.jl and InfiniteOpt.jl.

## Cart-Pole Optimal Control

The dynamic equation of the motion of cart-pole swing-up problem are given by:

$$

# \begin{bmatrix}
\ddot{x} \\
\ddot{\theta}
\end{bmatrix}

\begin{bmatrix}
\cos\theta & \ell \\
m_1 + m_2 & m_2 \ell \cos\theta
\end{bmatrix}^{-1}
\begin{bmatrix}

  - g \sin\theta \\
    F + m_2 \ell \dot{\theta}^2 \sin\theta
    \end{bmatrix}
    $$

where $x$ is the location of the cart, $\theta$ is the pole angle, $m_1$ is the cart mass, $m_2$ is the pole mass, $l$ is the pole length.

By converting the dynamics to first order equations, we can get the formulation:

# $$
\begin{bmatrix}
\dot{x} \\
\dot{\theta} \\
\ddot{x} \\
\ddot{\theta} \\
\dot{e}
\end{bmatrix}

# f\!\left(
\begin{bmatrix}
x \\ \theta \\ \dot{x} \\ \dot{\theta} \\ e
\end{bmatrix}
\right)

\begin{bmatrix}
\dot{x} \\
\dot{\theta} \\
\dfrac{-m_2 g \sin\theta \cos\theta - \left(F + m_2 \ell \dot{\theta}^2 \sin\theta\right)}
{m_2 \cos^2\theta - (m_1 + m_2)} \\
\dfrac{(m_1 + m_2) g \sin\theta + \cos\theta \left(F + m_2 \ell \dot{\theta}^2 \sin\theta\right)}
{m_2 \ell \cos^2\theta - (m_1 + m_2)\ell} \\
F^2
\end{bmatrix}
$$

and the initial conditions of all states at $t=0$ are all zero, the boundary conditions at time $t_f$ are:

$$
x_f=d, \dot{x_f}=0, \theta_f=\pi, \dot{\theta_f}=0
$$

The target cost function is defined as the "energy"

```julia
using BoundaryValueDiffEqMIRK, OptimizationMOI, Ipopt, Plots
m_1 = 1.0                      # Cart mass
m_2 = 0.3                      # Pole mass
l = 0.5                        # Pole length
d = 2.0                        # Cart target location
t_0 = 0.0                      # Start time
t_f = 2.0                      # Final time
g = 9.81                       # Gravity constant
tspan = (0.0, t_f)
function cart_pole!(du, u, p, t)
    x, θ, dx, dθ, f = u[1], u[2], u[3], u[4], u[5]
    du[1] = dx
    du[2] = dθ
    du[3] = (- m_2*g*sin(θ)*cos(θ) - (f + m_2*l*θ^2*sin(θ))) / (m_2*l*cos(θ)^2 - m_1 - m_2)
    du[4] = ((m_1 + m_2)*g*sin(θ) + cos(θ)*(f + m_1*l*dθ^2*sin(θ))) /
            (m_2*l*cos(θ)^2 - (m_1 + m_2)*l)
end

function cart_pole_bc!(du, u, p, t)
    du[1] = u(t_f)[1] - d
    du[2] = u(t_f)[2] - π
    du[3] = u(t_0)[2] - 0.0
    du[4] = u(t_0)[3] - 0.0
    du[5] = u(t_0)[4] - 0.0
    du[6] = u(t_f)[3] - 0.0
    du[7] = u(t_f)[4] - 0.0
end
cost_fun(u, p) = u[end]
u0 = [0.0, 0.0, 0.0, 0.0, 10.0]
cart_pole_fun = BVPFunction(cart_pole!, cart_pole_bc!; cost = cost_fun,
    bcresid_prototype = zeros(7), f_prototype = zeros(4))
cart_pole_prob = BVProblem(cart_pole_fun, u0, tspan; lb = [-2.0, -Inf, -Inf, -Inf, -20.0],
    ub = [2.0, Inf, Inf, Inf, 20.0])
sol = solve(cart_pole_prob, MIRK4(; optimize = Ipopt.Optimizer()); dt = 0.01, adaptive = false)

t = sol.t
x, theta, dx, dtheta, f = sol[1, :], sol[2, :], sol[3, :], sol[4, :], sol[5, :]

L = 1.0    # pole length (visual)
cart_w = 0.4    # cart width
cart_h = 0.2    # cart height

# Precompute pole tip coordinates
px = x .+ L .* sin.(theta)
py = cart_h/2 .- L .* cos.(theta)

# Axis limits (a bit margin around the cart trajectory)
xmin = minimum(x) - 2L
xmax = maximum(x) + 2L
ymin = -2.0
ymax = 2.0

anim = @animate for k in eachindex(t)
    cart_x = x[k]
    pole_x = px[k]
    pole_y = py[k]

    # Base plot / axis
    plot(; xlim = (xmin, xmax), ylim = (ymin, ymax), aspect_ratio = :equal,
        legend = false, title = "Cart–Pole (t = $(round(t[k], digits=2)) s)")

    # Draw ground
    plot!([xmin, xmax], [0, 0], lw = 2, color = :black)

    # Draw cart as a rectangle
    rect = Shape(
        [cart_x - cart_w/2, cart_x + cart_w/2, cart_x + cart_w/2, cart_x - cart_w/2],
        [0, 0, cart_h, cart_h])
    plot!(rect, color = :gray)

    # Draw pole as a line from cart center to tip
    plot!([cart_x, pole_x], [cart_h/2, pole_y], lw = 3, color = :red)

    # Draw pivot point
    scatter!([cart_x], [cart_h/2], ms = 4, color = :black)
end

# Save GIF
gif(anim, "./cart_pole.gif", fps = 40)
```

![cart_pole](./cart_pole.gif)
