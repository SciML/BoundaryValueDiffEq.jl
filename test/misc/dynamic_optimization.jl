@testitem "Rocket launch" begin
    using BoundaryValueDiffEq
    using OptimizationMOI, Ipopt
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
    D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    g(x_h) = g_0 * (h_0 / x_h)^2
    function rocket_launch!(du, u, p, t)
        # u_t is the control variable (thrust)
        x_v, x_h, x_m, u_t = u[1], u[2], u[3], u[4]
        du[1] = (u_t-drag(x_h, x_v))/x_m - g(x_h)
        du[2] = x_v
        du[3] = -u_t/c
    end
    function constraints!(res, u, p)
        res[1] = u[1]
        res[2] = u[2]
        res[3] = u[3]
        res[4] = u[4]
    end
    cost_fun(u, p) = -u[4]
    u0 = [v_0, h_0, m_0, 0.0]
    rocket_launch_fun = BVPFunction(rocket_launch!, inequality = constraints!)
    rocket_launch_prob = BVProblem(rocket_launch_fun, u0, tspan; cost = cost_fun,
        lcons = [0.0, 0.0, m_T, 0.0], ucons = [Inf, Inf, Inf, u_t_max])
end
