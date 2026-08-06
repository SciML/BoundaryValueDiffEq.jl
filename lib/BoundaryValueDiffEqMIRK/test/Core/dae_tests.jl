using BoundaryValueDiffEqMIRK
using Test

# Index-1 DAE BVPs solved via unprojected collocation (Ascher & Spiteri 1994).
# The algebraic constraint is enforced exactly at the mesh points, so accuracy for
# algebraic variables is only checked there: the continuous interpolant is not
# accurate for algebraic components, which is also why mesh adaptivity is not
# supported for DAEs (the defect estimate cannot converge).

@testset "Simple index-1 DAE" begin
    using BoundaryValueDiffEqMIRK, SciMLBase
    using LinearAlgebra

    # u1' = u2, 0 = u2 - cos(t) with u1(0) = 0
    # Analytic solution: u1 = sin(t), u2 = cos(t)
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = u[2] - cos(t)
    end
    f1(u, p, t) = [u[2], u[2] - cos(t)]
    function bc1!(res, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(0.0)[2] - 1.0
    end
    bc1(u, p, t) = [u(0.0)[1], u(0.0)[2] - 1.0]

    mass_matrix = [1.0 0.0; 0.0 0.0]
    tspan = (0.0, pi / 2)
    prob_iip = BVProblem(BVPFunction(f1!, bc1!; mass_matrix), [0.0, 1.0], tspan)
    prob_oop = BVProblem(BVPFunction(f1, bc1; mass_matrix), [0.0, 1.0], tspan)

    @testset "$(nameof(typeof(alg))), $(SciMLBase.isinplace(prob) ? "iip" : "oop")" for alg in
            (MIRK2(), MIRK3(), MIRK4(), MIRK5(), MIRK6()),
            prob in (prob_iip, prob_oop)

        sol = solve(prob, alg; dt = 0.01, adaptive = false)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs(sol.u[i][1] - sin(sol.t[i])) for i in eachindex(sol.t)) < 5.0e-3
        @test maximum(abs(sol.u[i][2] - cos(sol.t[i])) for i in eachindex(sol.t)) < 1.0e-12
    end
end

@testset "Ascher & Spiteri example problem 1" begin
    using BoundaryValueDiffEqMIRK, SciMLBase
    using LinearAlgebra

    # Singular index-1 BVDAE from the Ascher & Spiteri paper.
    # Analytic solution: [sin(t), sin(t), 1, 0]
    function f2!(du, u, p, t)
        e = 2.7
        du[1] = (1 + u[2] - sin(t)) * u[4] + cos(t)
        du[2] = cos(t)
        du[3] = u[4]
        du[4] = (u[1] - sin(t)) * (u[4] - e^t)
    end
    function f2(u, p, t)
        e = 2.7
        return [
            (1 + u[2] - sin(t)) * u[4] + cos(t), cos(t),
            u[4], (u[1] - sin(t)) * (u[4] - e^t),
        ]
    end
    function bc2!(res, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(0.0)[3] - 1.0
        res[3] = u(1.0)[2] - sin(1.0)
        res[4] = u(0.0)[4]
    end
    bc2(u, p, t) = [u(0.0)[1], u(0.0)[3] - 1.0, u(1.0)[2] - sin(1.0), u(0.0)[4]]
    f2_analytic(t) = [sin(t), sin(t), 1.0, 0.0]

    mass_matrix = [
        1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0;
        0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0
    ]
    tspan = (0.0, 1.0)
    prob_iip = BVProblem(BVPFunction(f2!, bc2!; mass_matrix), zeros(4), tspan)
    prob_oop = BVProblem(BVPFunction(f2, bc2; mass_matrix), zeros(4), tspan)

    @testset "$(nameof(typeof(alg))), $(SciMLBase.isinplace(prob) ? "iip" : "oop")" for alg in
            (MIRK4(), MIRK6()),
            prob in (prob_iip, prob_oop)

        sol = solve(prob, alg; dt = 0.01, adaptive = false)
        @test SciMLBase.successful_retcode(sol)
        err = maximum(
            maximum(abs.(sol.u[i] .- f2_analytic(sol.t[i]))) for i in eachindex(sol.t)
        )
        @test err < 1.0e-6
    end
end

@testset "Mesh adaptivity is not supported for DAEs" begin
    using BoundaryValueDiffEqMIRK, SciMLBase
    using LinearAlgebra

    function f3!(du, u, p, t)
        du[1] = u[2]
        du[2] = u[2] - cos(t)
    end
    function bc3!(res, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(0.0)[2] - 1.0
    end
    mass_matrix = [1.0 0.0; 0.0 0.0]
    prob = BVProblem(BVPFunction(f3!, bc3!; mass_matrix), [0.0, 1.0], (0.0, pi / 2))

    @test_throws ArgumentError solve(prob, MIRK4(); dt = 0.05)
    @test_throws ArgumentError solve(prob, MIRK4(); dt = 0.05, adaptive = true)
end
