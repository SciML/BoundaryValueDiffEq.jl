@testitem "Type Stability" begin
    using LinearAlgebra, BoundaryValueDiffEq, OrdinaryDiffEq

    f(u, p, t) = [p[1] * u[1] - p[2] * u[1] * u[2], p[3] * u[1] * u[2] - p[4] * u[2]]
    function f!(du, u, p, t)
        du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
        du[2] = p[3] * u[1] * u[2] - p[4] * u[2]
    end

    bc(sol, p, t) = [sol(0.0)[1] - 1, sol(1.0)[2] - 2]
    function bc!(res, sol, p, t)
        res[1] = sol(0.0)[1] - 1
        res[2] = sol(1.0)[2] - 2
    end
    twobc_a(ua, p) = [ua[1] - 1]
    twobc_b(ub, p) = [ub[2] - 2]
    twobc_a!(resa, ua, p) = (resa[1] = ua[1] - 1)
    twobc_b!(resb, ub, p) = (resb[1] = ub[2] - 2)

    u0 = Float64[0, 0]
    tspan = (0.0, 1.0)
    p = [1.0, 1.0, 1.0, 1.0]
    bcresid_prototype = (zeros(1), zeros(1))

    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    # Multi-Point BVP
    @testset "Multi-Point BVP" begin
        mpbvp_iip = BVProblem(f!, bc!, u0, tspan, p; nlls = Val(false))
        mpbvp_oop = BVProblem(f, bc, u0, tspan, p; nlls = Val(false))

        @testset "Shooting Methods" begin
            @inferred solve(mpbvp_iip, Shooting(Tsit5(); jac_alg))
            @inferred solve(mpbvp_oop, Shooting(Tsit5(); jac_alg))
            @inferred solve(mpbvp_iip, MultipleShooting(5, Tsit5(); jac_alg))
            @inferred solve(mpbvp_oop, MultipleShooting(5, Tsit5(); jac_alg))
        end

        @testset "MIRK Methods" begin
            for solver in (MIRK2(; jac_alg), MIRK3(; jac_alg), MIRK4(; jac_alg),
                MIRK5(; jac_alg), MIRK6(; jac_alg))
                @inferred solve(mpbvp_iip, solver; dt = 0.2)
                @inferred solve(mpbvp_oop, solver; dt = 0.2)
            end
        end
    end

    # Two-Point BVP
    @testset "Two-Point BVP" begin
        tpbvp_iip = TwoPointBVProblem(
            f!, (twobc_a!, twobc_b!), u0, tspan, p; bcresid_prototype, nlls = Val(false))
        tpbvp_oop = TwoPointBVProblem(
            f, (twobc_a, twobc_b), u0, tspan, p; nlls = Val(false))

        @testset "Shooting Methods" begin
            @inferred solve(tpbvp_iip, Shooting(Tsit5(); jac_alg))
            @inferred solve(tpbvp_oop, Shooting(Tsit5(); jac_alg))
            @inferred solve(tpbvp_iip, MultipleShooting(5, Tsit5(); jac_alg))
            @inferred solve(tpbvp_oop, MultipleShooting(5, Tsit5(); jac_alg))
        end

        @testset "MIRK Methods" begin
            for solver in (MIRK2(; jac_alg), MIRK3(; jac_alg), MIRK4(; jac_alg),
                MIRK5(; jac_alg), MIRK6(; jac_alg))
                @inferred solve(tpbvp_iip, solver; dt = 0.2)
                @inferred solve(tpbvp_oop, solver; dt = 0.2)
            end
        end
    end
end
