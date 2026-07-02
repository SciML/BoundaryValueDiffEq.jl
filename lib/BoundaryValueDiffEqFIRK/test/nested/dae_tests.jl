using BoundaryValueDiffEqFIRK
using Test

# Index-1 DAE BVPs solved via unprojected collocation (Ascher & Spiteri 1994).
# The algebraic constraint is enforced exactly at the mesh points, so accuracy for
# algebraic variables is only checked there: the continuous interpolant is not
# accurate for algebraic components, which is also why mesh adaptivity is not
# supported for DAEs (the defect estimate cannot converge).
#
# LobattoIIIa and LobattoIIIb are excluded: their tableau structure leaves the
# algebraic components of the stages underdetermined, so they cannot solve DAEs.
# Problems whose constraint Jacobian is singular on the solution (e.g. the Ascher
# & Spiteri example problem 1) are also excluded: the nested nonlinear solve of
# the stage equations hits the singular constraint directly.

@testset "Simple index-1 DAE" begin
    using BoundaryValueDiffEqFIRK, SciMLBase
    using LinearAlgebra

    nested = true

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
            (
                RadauIIa3(; nested_nlsolve = nested), RadauIIa5(; nested_nlsolve = nested),
                LobattoIIIc4(; nested_nlsolve = nested),
            ),
            prob in (prob_iip, prob_oop)

        sol = solve(prob, alg; dt = 0.01, adaptive = false)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs(sol.u[i][1] - sin(sol.t[i])) for i in eachindex(sol.t)) < 1.0e-10
        @test maximum(abs(sol.u[i][2] - cos(sol.t[i])) for i in eachindex(sol.t)) < 1.0e-12
    end
end

@testset "Mesh adaptivity is not supported for DAEs" begin
    using BoundaryValueDiffEqFIRK, SciMLBase
    using LinearAlgebra

    nested = true

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

    @test_throws ArgumentError solve(prob, RadauIIa5(; nested_nlsolve = nested); dt = 0.05)
    @test_throws ArgumentError solve(
        prob, RadauIIa5(; nested_nlsolve = nested); dt = 0.05, adaptive = true
    )
end
