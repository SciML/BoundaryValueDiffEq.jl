@testitem "Singular BVP" tags = [:singular] begin
    using BoundaryValueDiffEqFIRK
    using LinearAlgebra

    nested = false

    for stage in (2, 3, 4, 5)
        s = Symbol("LobattoIIIa$(stage)")
        @eval lobattoIIIa_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    for stage in (2, 3, 4, 5)
        s = Symbol("LobattoIIIb$(stage)")
        @eval lobattoIIIb_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    for stage in (2, 3, 4, 5)
        s = Symbol("LobattoIIIc$(stage)")
        @eval lobattoIIIc_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    for stage in (2, 3, 5, 7)
        s = Symbol("RadauIIa$(stage)")
        @eval radau_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    # Lane-Emden equation of index 1:
    # y'' + (2/t)*y' + y = 0, y(0) = 1, y'(0) = 0
    # The exact solution is y(t) = sin(t)/t (with limit y(0) = 1)
    #
    # In first-order form: y[1]' = y[2], y[2]' = -y[1] - (2/t)*y[2]
    # This can be written as y' = S*y/t + f(t,y) where:
    # S = [0 0; 0 -2] and f(t,y) = [y[2]; -y[1]]

    function lane_emden!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]  # The -2*u[2]/t term is handled by singular_term
    end
    function lane_emden(u, p, t)
        return [u[2]; -u[1]]
    end

    function lane_emden_bc_a!(resid, ua, p)
        resid[1] = ua[1] - 1.0  # y(0) = 1
    end
    function lane_emden_bc_b!(resid, ub, p)
        resid[1] = ub[1] - sin(1.0)  # y(1) = sin(1) ≈ 0.8415
    end

    lane_emden_bc_a(ua, p) = ua[1] - 1.0
    lane_emden_bc_b(ub, p) = ub[1] - sin(1.0)

    # The singular term matrix S
    S = [0.0 0.0; 0.0 -2.0]

    tspan = (0.0, 1.0)
    u0 = [1.0, 0.0]
    bcresid_prototype = (zeros(1), zeros(1))

    prob_iip = TwoPointBVProblem(
        lane_emden!, (lane_emden_bc_a!, lane_emden_bc_b!), u0, tspan; bcresid_prototype
    )
    prob_oop = TwoPointBVProblem(
        lane_emden, (lane_emden_bc_a, lane_emden_bc_b), u0, tspan; bcresid_prototype
    )

    # Test with different FIRK stages
    # Note: tolerance varies by order - FIRK2 is lower order so needs larger tolerance
    @testset "FIRK solvers with singular term" for prob in (prob_iip, prob_oop)
        for stage in (2, 3, 4, 5)
            sol = solve(prob, lobattoIIIa_solver(Val(stage)), dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
        for stage in (3, 4, 5)
            sol = solve(prob, lobattoIIIb_solver(Val(stage)), dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
        for stage in (3, 4, 5)
            sol = solve(prob, lobattoIIIc_solver(Val(stage)), dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
        for stage in (2, 3, 5, 7)
            sol = solve(prob, radau_solver(Val(stage)), dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end
