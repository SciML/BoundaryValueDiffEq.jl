using BoundaryValueDiffEq, OrdinaryDiffEq, LinearAlgebra, Test

f(u, p, t) = [p[1] * u[1] - p[2] * u[1] * u[2], p[3] * u[1] * u[2] - p[4] * u[2]]
function f!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] * u[2] - p[4] * u[2]
end

bc(sol, p, t) = [sol[1][1] - 1, sol[end][2] - 2]
function bc!(res, sol, p, t)
    res[1] = sol[1][1] - 1
    res[2] = sol[end][2] - 2
end
twobc_a(ua, p) = [ua[1] - 1]
twobc_b(ub, p) = [ub[2] - 2]
twobc_a!(resa, ua, p) = (resa[1] = ua[1] - 1)
twobc_b!(resb, ub, p) = (resb[1] = ub[2] - 2)

u0 = Float64[0, 0]
tspan = (0.0, 1.0)
p = [1.0, 1.0, 1.0, 1.0]
bcresid_prototype = (zeros(1), zeros(1))

# Multi-Point BVP
@testset "Multi-Point BVP" begin
    mpbvp_iip = BVProblem(f!, bc!, u0, tspan, p)
    mpbvp_oop = BVProblem(f, bc, u0, tspan, p)

    @testset "Shooting Methods" begin
        @inferred solve(mpbvp_iip, Shooting(Tsit5()))
        @inferred solve(mpbvp_oop, Shooting(Tsit5()))
        @inferred solve(mpbvp_iip, MultipleShooting(5, Tsit5()))
        @inferred solve(mpbvp_oop, MultipleShooting(5, Tsit5()))
    end

    @testset "MIRK Methods" begin
        for solver in (MIRK2(), MIRK3(), MIRK4(), MIRK5(), MIRK6())
            @inferred solve(mpbvp_iip, solver; dt = 0.2)
            @inferred solve(mpbvp_oop, solver; dt = 0.2)
        end
    end
end

# Two-Point BVP
@testset "Two-Point BVP" begin
    tpbvp_iip = TwoPointBVProblem(f!, (twobc_a!, twobc_b!), u0, tspan, p; bcresid_prototype)
    tpbvp_oop = TwoPointBVProblem(f, (twobc_a, twobc_b), u0, tspan, p)

    @testset "Shooting Methods" begin
        @inferred solve(tpbvp_iip, Shooting(Tsit5()))
        @inferred solve(tpbvp_oop, Shooting(Tsit5()))
        @inferred solve(tpbvp_iip, MultipleShooting(5, Tsit5()))
        @inferred solve(tpbvp_oop, MultipleShooting(5, Tsit5()))
    end

    @testset "MIRK Methods" begin
        for solver in (MIRK2(), MIRK3(), MIRK4(), MIRK5(), MIRK6())
            @inferred solve(tpbvp_iip, solver; dt = 0.2)
            @inferred solve(tpbvp_oop, solver; dt = 0.2)
        end
    end
end
