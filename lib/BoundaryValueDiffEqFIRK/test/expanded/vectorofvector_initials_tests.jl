@testitem "VectorOfVector Initial Condition" begin
    using BoundaryValueDiffEqFIRK, OrdinaryDiffEq
    #System Constants
    ss = 1 #excitatory parameter
    sj = 0 #inhibitory parameter
    glb = 0.05
    el = -70
    gnab = 3
    ena = 50
    gkb = 5
    ek = -90
    gtb = 2
    et = 90
    gex = 0.1
    vex = 0
    gsyn = 0.13
    vsyn = -85
    iext = 0.41
    eps = 1
    qht = 2.5

    #System functions
    function f(v, h, r)
        -(glb * (v - el) +
          gnab * (1 / (1 + exp(-(v + 37) / 7)))^3 * h * (v - ena) +
          gkb * (0.75 * (1 - h))^4 * (v - ek) +
          gtb * (1 / (1 + exp(-(v + 60) / 6.2)))^2 * r * (v - et)) - gex * ss * (v - vex) -
        gsyn * sj * (v - vsyn) + iext
    end

    function g(v, h)
        eps * ((1 / (1 + exp((v + 41) / 4))) - h) /
        (1 / ((0.128 * exp(-(v + 46) / 18)) + (4 / (1 + exp(-(v + 23) / 5)))))
    end

    function k(v, r)
        qht * ((1 / (1 + exp((v + 84) / 4))) - r) / ((28 + exp(-(v + 25) / 10.5)))
    end

    #Dynamical System
    function TC!(du, u, p, t)
        v, h, r = u

        du[1] = dv = f(v, h, r)
        du[2] = dh = g(v, h)
        du[3] = dr = k(v, r)
    end

    #Finding initial guesses by forward integration
    T = 7.588145762136627 #orbit length
    u0 = [-40.296570996984855, 0.7298857398191566, 0.0011680534089275774]
    tspan = (0.0, T)
    prob = ODEProblem(TC!, u0, tspan, dt = 0.01)
    sol = solve(prob, Rodas4P(), reltol = 1e-12, abstol = 1e-12, saveat = 0.5)

    # The BVP set up
    # This is not really kind of Two-Point BVP we support.
    function bc_po!(residual, u, p, t)
        residual[1] = u(0.0)[1] - u(T)[1]
        residual[2] = u(0.0)[2] - u(T)[2]
        residual[3] = u(0.0)[3] - u(T)[3]
    end

    #This is the part of the code that has problems
    bvp1 = BVProblem(TC!, bc_po!, sol.u, tspan)
    sol5 = solve(bvp1, RadauIIa5(); dt = 0.5)
    @test SciMLBase.successful_retcode(sol5.retcode)

    bvp1 = BVProblem(TC!, bc_po!, zero(first(sol.u)), tspan)
    sol5 = solve(bvp1, RadauIIa5(); dt = 0.1, abstol = 1e-14)
    @test SciMLBase.successful_retcode(sol5.retcode)
end
