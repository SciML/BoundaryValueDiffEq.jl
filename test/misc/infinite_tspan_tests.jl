@testitem "Infinite Time Span BVP" tags=[:mirk] begin
    using BoundaryValueDiffEq

    # Test 1: Two-point BVP on [0, ∞) with exponential decay
    # ODE: u' = -u with u(0) = 1, u(∞) ≈ 0
    # Analytical solution: u(t) = exp(-t)
    @testset "TwoPointBVP exponential decay" begin
        function f!(du, u, p, t)
            du[1] = -u[1]
        end

        bc_a! = (residual, ua, p) -> (residual[1] = ua[1] - 1.0)
        bc_b! = (residual, ub, p) -> (residual[1] = ub[1])

        # Use initial guess function that provides a reasonable starting point
        initial_guess = (p, t) -> [exp(-t)]

        tspan = (0.0, Inf)
        bcresid_prototype = (zeros(1), zeros(1))

        prob = TwoPointBVProblem(f!, (bc_a!, bc_b!), initial_guess, tspan; bcresid_prototype)

        # Use non-adaptive mode for faster convergence
        sol = solve(prob, MIRK4(); dt = 0.1, adaptive = false)
        @test SciMLBase.successful_retcode(sol)

        # Check boundary conditions are satisfied
        @test sol(0.0)[1] ≈ 1.0 atol = 1e-3
        @test sol(1.0)[1] ≈ exp(-1.0) atol = 0.1
    end

    # Test 2: Scalar ODE with initial guess
    @testset "Exponential decay with initial guess" begin
        function f!(du, u, p, t)
            du[1] = -u[1]
        end

        bc_a! = (residual, ua, p) -> (residual[1] = ua[1] - 1.0)
        bc_b! = (residual, ub, p) -> (residual[1] = ub[1])

        # Constant initial guess
        u0 = [0.5]
        tspan = (0.0, Inf)
        bcresid_prototype = (zeros(1), zeros(1))

        prob = TwoPointBVProblem(f!, (bc_a!, bc_b!), u0, tspan; bcresid_prototype)

        sol = solve(prob, MIRK4(); dt = 0.1, adaptive = false)
        @test SciMLBase.successful_retcode(sol)

        # Initial condition should be satisfied
        @test isapprox(sol(0.0)[1], 1.0, atol = 0.1)
    end
end

@testitem "Time Domain Transformations" tags=[:core] begin
    using BoundaryValueDiffEqCore

    @testset "IdentityTransform" begin
        trans = IdentityTransform()
        @test τ_to_t(trans, 0.5) == 0.5
        @test t_to_τ(trans, 0.5) == 0.5
        @test dtdτ(trans, 0.5) == 1.0
        @test is_identity_transform(trans) == true
    end

    @testset "SemiInfiniteTransform" begin
        trans = SemiInfiniteTransform(0.0)

        # Test τ = 0 corresponds to t = 0
        @test τ_to_t(trans, 0.0) ≈ 0.0
        @test t_to_τ(trans, 0.0) ≈ 0.0

        # Test τ = 1 corresponds to t = ∞
        @test isinf(τ_to_t(trans, 1.0))
        @test t_to_τ(trans, Inf) ≈ 1.0

        # Test intermediate values
        # τ = 0.5 → t = 0 + 0.5/(1 - 0.5) = 1.0
        @test τ_to_t(trans, 0.5) ≈ 1.0
        @test t_to_τ(trans, 1.0) ≈ 0.5

        # Test dt/dτ = 1/(1 - τ)²
        @test dtdτ(trans, 0.0) ≈ 1.0
        @test dtdτ(trans, 0.5) ≈ 4.0  # 1/(0.5)² = 4

        @test is_identity_transform(trans) == false
    end

    @testset "SemiInfiniteTransform with offset" begin
        trans = SemiInfiniteTransform(2.0)

        # Test τ = 0 corresponds to t = 2
        @test τ_to_t(trans, 0.0) ≈ 2.0
        @test t_to_τ(trans, 2.0) ≈ 0.0

        # Test τ = 0.5 → t = 2 + 0.5/(1 - 0.5) = 3.0
        @test τ_to_t(trans, 0.5) ≈ 3.0
        @test t_to_τ(trans, 3.0) ≈ 0.5
    end

    @testset "select_transform" begin
        # Finite interval
        trans, τ₀, τ₁ = select_transform(0.0, 10.0)
        @test is_identity_transform(trans)
        @test τ₀ == 0.0
        @test τ₁ == 10.0

        # Semi-infinite interval [0, ∞)
        trans, τ₀, τ₁ = select_transform(0.0, Inf)
        @test trans isa SemiInfiniteTransform
        @test τ₀ == 0.0
        @test τ₁ == 1.0

        # Semi-infinite interval [5, ∞)
        trans, τ₀, τ₁ = select_transform(5.0, Inf)
        @test trans isa SemiInfiniteTransform
        @test trans.a == 5.0
    end
end
