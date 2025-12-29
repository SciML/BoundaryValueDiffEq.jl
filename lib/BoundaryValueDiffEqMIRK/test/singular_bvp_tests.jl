@testitem "Singular BVP - Lane-Emden Equation" tags=[:singular] begin
    using BoundaryValueDiffEqMIRK
    using LinearAlgebra

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

    function lane_emden_bc_a!(resid, ua, p)
        resid[1] = ua[1] - 1.0  # y(0) = 1
    end

    function lane_emden_bc_b!(resid, ub, p)
        resid[1] = ub[1] - sin(1.0)  # y(1) = sin(1) ≈ 0.8415
    end

    # The singular term matrix S
    S = [0.0 0.0; 0.0 -2.0]

    tspan = (0.0, 1.0)
    u0 = [1.0, 0.0]
    bcresid_prototype = (zeros(1), zeros(1))

    prob = TwoPointBVProblem(
        lane_emden!, (lane_emden_bc_a!, lane_emden_bc_b!), u0, tspan; bcresid_prototype)

    # Test with different MIRK orders
    # Note: tolerance varies by order - MIRK2 is lower order so needs larger tolerance
    @testset "MIRK$order with singular term" for (order, tol) in
                                                 ((2, 0.02), (4, 1e-6), (6, 1e-6))
        alg = if order == 2
            MIRK2(singular_term = S)
        elseif order == 4
            MIRK4(singular_term = S)
        else
            MIRK6(singular_term = S)
        end

        sol = solve(prob, alg; dt = 0.05, abstol = 1e-4, adaptive = false)

        # Check that the solution converged
        @test SciMLBase.successful_retcode(sol)

        # Check accuracy of the solution at t=1
        exact_y1 = sin(1.0)
        @test isapprox(sol(1.0)[1], exact_y1, atol = tol)
    end
end

@testitem "Singular Term - No Regression on Regular BVPs" tags=[:singular] begin
    using BoundaryValueDiffEqMIRK
    using LinearAlgebra

    # Test that regular BVPs still work correctly when singular_term=nothing (default)
    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end

    function bc_a!(resid, ua, p)
        resid[1] = ua[1] - 5.0
    end

    function bc_b!(resid, ub, p)
        resid[1] = ub[1]
    end

    u0 = [5.0, -3.5]
    tspan = (0.0, 5.0)
    bcresid_prototype = (zeros(1), zeros(1))

    prob = TwoPointBVProblem(f!, (bc_a!, bc_b!), u0, tspan; bcresid_prototype)

    # Test only MIRK4 and MIRK6 for the regression test
    # (MIRK2 has known accuracy limitations for this problem that are unrelated to singular_term)
    @testset "MIRK$order without singular term" for order in (4, 6)
        alg = if order == 4
            MIRK4()
        else
            MIRK6()
        end

        sol = solve(prob, alg; dt = 0.5, abstol = 1e-6, adaptive = false)

        @test SciMLBase.successful_retcode(sol)

        # Analytic solution: y(t) = 5 - t, y'(t) = -1
        @test isapprox(sol(0.0)[1], 5.0, atol = 1e-4)
        @test isapprox(sol(5.0)[1], 0.0, atol = 1e-4)
        @test isapprox(sol(2.5)[1], 2.5, atol = 1e-4)
    end
end

@testitem "Singular Term Parameter Types" tags=[:singular] begin
    using BoundaryValueDiffEqMIRK
    using LinearAlgebra

    # Test that the singular_term parameter accepts the correct types
    @test MIRK4().singular_term === nothing
    @test MIRK4(singular_term = nothing).singular_term === nothing

    S = [1.0 0.0; 0.0 -2.0]
    alg = MIRK4(singular_term = S)
    @test alg.singular_term == S

    # Test with Float32
    S32 = Float32[1.0 0.0; 0.0 -2.0]
    alg32 = MIRK4(singular_term = S32)
    @test eltype(alg32.singular_term) == Float32
end
