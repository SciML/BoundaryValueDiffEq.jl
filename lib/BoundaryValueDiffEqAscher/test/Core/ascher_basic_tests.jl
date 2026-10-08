using BoundaryValueDiffEqAscher
using Test

@testset "Public API" begin
    # Only the names this package owns; the rest of `names` is the reexported API
    # documented elsewhere, pinned against `ASCHER_REEXPORTS` by test/qa/qa.jl.
    owned = filter(names(BoundaryValueDiffEqAscher)) do name
        which(BoundaryValueDiffEqAscher, name) === BoundaryValueDiffEqAscher
    end
    @test Set(owned) == Set(
        (
            :Ascher1,
            :Ascher2,
            :Ascher3,
            :Ascher4,
            :Ascher5,
            :Ascher6,
            :Ascher7,
            :BoundaryValueDiffEqAscher,
        )
    )
end

# Standard test BVDAE problem from the URI M. ASCHER and RAYMOND J. SPITERI paper
@testset "Test Ascher solver on example problem 1" begin
    using BoundaryValueDiffEqAscher, SciMLBase
    function f1!(du, u, p, t)
        e = 2.7
        du[1] = (1 + u[2] - sin(t)) * u[4] + cos(t)
        du[2] = cos(t)
        du[3] = u[4]
        du[4] = (u[1] - sin(t)) * (u[4] - e^t)
    end
    function f1(u, p, t)
        e = 2.7
        return [
            (1 + u[2] - sin(t)) * u[4] + cos(t), cos(t),
            u[4], (u[1] - sin(t)) * (u[4] - e^t),
        ]
    end
    function bc1!(res, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(0.0)[3] - 1
        res[3] = u(1.0)[2] - sin(1.0)
    end
    function bc1(u, p, t)
        return [u(0.0)[1], u(0.0)[3] - 1, u(1.0)[2] - sin(1.0)]
    end
    function bca1!(res, ua, p)
        res[1] = ua[1]
        res[2] = ua[3] - 1
    end
    function bcb1!(res, ub, p)
        res[1] = ub[2] - sin(1.0)
    end
    function bca1(ua, p)
        return [ua[1], ua[3] - 1]
    end
    function bcb1(ub, p)
        return [ub[2] - sin(1.0)]
    end
    function f1_analytic(u, p, t)
        return [sin(t), sin(t), 1.0, 0.0]
    end
    u01 = [0.0, 0.0, 0.0, 0.0]
    tspan1 = (0.0, 1.0)
    fun_iip = ODEFunction(
        f1!, analytic = f1_analytic, mass_matrix = [
            1 0 0 0; 0 1 0 0;
            0 0 1 0; 0 0 0 0
        ]
    )
    fun_oop = ODEFunction(
        f1, analytic = f1_analytic, mass_matrix = [
            1 0 0 0; 0 1 0 0;
            0 0 1 0; 0 0 0 0
        ]
    )
    prob_iip = BVProblem(fun_iip, bc1!, u01, tspan1)
    prob_oop = BVProblem(fun_oop, bc1, u01, tspan1)
    tpprob_iip = TwoPointBVProblem(
        fun_iip, (bca1!, bcb1!), u01, tspan1, bcresid_prototype = (zeros(2), zeros(1))
    )
    tpprob_oop = TwoPointBVProblem(
        fun_oop, (bca1, bcb1), u01, tspan1, bcresid_prototype = (zeros(2), zeros(1))
    )
    prob1Arr = [prob_iip, prob_oop, tpprob_iip, tpprob_oop]
    SOLVERS = [
        alg()
            for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)
    ]
    for i in 1:4
        for stage in (3, 4, 5, 6, 7)
            sol = solve(prob1Arr[i], SOLVERS[stage], dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
            @test sol.errors[:final] < 1.0e-4
        end
    end
end

### Another BVDAE problem ###
# Comes from "Boundary value problems for differential-algebraic equations"
# by Leonid V. Kalachev and Robert E. O'Malley
@testset "Test Ascher solver on example problem 2" begin
    using BoundaryValueDiffEqAscher, SciMLBase

    function f2!(du, u, p, t)
        du[1] = u[2] + u[3] + u[5] + 1
        du[2] = u[2] + u[4]
        du[3] = u[1] + u[5]
        du[4] = u[1] + u[2] + 1
        du[5] = u[1] + u[3]
    end

    function f2(u, p, t)
        return [
            u[2] + u[3] + u[5] + 1, u[2] + u[4], u[1] + u[5], u[1] + u[2] + 1, u[1] + u[3],
        ]
    end

    function bc2!(res, u, p, t)
        res[1] = u(0.0)[1] + 1
        res[2] = u(1.0)[2] + 2
        res[3] = u(1.0)[3] - 1
    end
    function bc2(u, p, t)
        return [u(0.0)[1] + 1, u(1.0)[2] + 2, u(1.0)[3] - 1]
    end
    u02 = [0.0, 0.0, 0.0, 0.0, 0.0]
    tspan2 = (0.0, 1.0)
    fun2_iip = BVPFunction(
        f2!, bc2!, mass_matrix = [
            1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0;
            0 0 0 0 0; 0 0 0 0 0
        ]
    )
    fun2_oop = BVPFunction(
        f2, bc2, mass_matrix = [
            1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0;
            0 0 0 0 0; 0 0 0 0 0
        ]
    )
    prob2_iip = BVProblem(fun2_iip, u02, tspan2)
    prob2_oop = BVProblem(fun2_oop, u02, tspan2)
    prob2Arr = [prob2_iip, prob2_oop]
    SOLVERS = [
        alg()
            for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)
    ]
    for i in 1:2
        for stage in (2, 4, 5, 6)
            sol = solve(prob2Arr[i], SOLVERS[stage], dt = 0.01, adaptive = false)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

@testset "Test Ascher solver on example problem 3" begin
    using BoundaryValueDiffEqAscher, SciMLBase
    function f3!(du, u, p, t)
        du[1] = -u[3]
        du[2] = -u[3]
        du[3] = u[2] - sin(t - 1)
    end
    function f3(u, p, t)
        return [-u[3], -u[3], u[2] - sin(t - 1)]
    end
    function bc3!(res, u, p, t)
        res[1] = u(1.0)[1]
        res[2] = u(1.0)[2]
    end
    function bc3(u, p, t)
        return [u(1.0)[1], u(1.0)[2]]
    end
    function f3_analytic(u, p, t)
        return [sin(t - 1), sin(t - 1), -cos(t - 1)]
    end
    u03 = [0.0, 0.0, 0.0]
    tspan3 = (0.0, 1.0)
    fun_iip = ODEFunction(f3!, analytic = f3_analytic, mass_matrix = [1 0 0; 0 1 0; 0 0 0])
    fun_oop = ODEFunction(f3, analytic = f3_analytic, mass_matrix = [1 0 0; 0 1 0; 0 0 0])
    prob_iip = BVProblem(fun_iip, bc3!, u03, tspan3)
    prob_oop = BVProblem(fun_oop, bc3, u03, tspan3)
    prob3Arr = [prob_iip, prob_oop]
    SOLVERS = [
        alg()
            for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)
    ]
    for i in 1:2
        for stage in (2, 3, 4, 5, 6, 7)
            sol = solve(prob3Arr[i], SOLVERS[stage], dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

# JET tests have been moved to the separate QA test group (test/qa/)

@testset "Two-point boundary allocation" begin
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = 0; nothing)
    f(u, p, t) = [u[2], 0.0]
    for nleft in 0:2
        bca(u, p) = [u[1], u[2] - 1][1:nleft]
        bcb(u, p) = [u[1] - 1, u[2] - 1][(nleft + 1):2]
        bca!(res, u, p) = (res .= bca(u, p); nothing)
        bcb!(res, u, p) = (res .= bcb(u, p); nothing)
        prototype = (zeros(nleft), zeros(2 - nleft))
        for prob in (
                    TwoPointBVProblem(f!, (bca!, bcb!), zeros(2), (0.0, 1.0); bcresid_prototype = prototype),
                    TwoPointBVProblem(f, (bca, bcb), zeros(2), (0.0, 1.0)),
                ), adaptive in (false, true)
            sol = solve(prob, Ascher2(); dt = 0.25, adaptive, abstol = 1.0e-8)
            @test successful_retcode(sol)
            @test sol.u[1] ≈ [0, 1] atol = 1.0e-7
            @test sol.u[end] ≈ [1, 1] atol = 1.0e-7
        end
    end
end

@testset "Interpolated boundary conditions" begin
    # Two interior evaluation points in one residual also exercises coupling
    # between intervals, which separated side conditions cannot represent.
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = 0; nothing)
    f(u, p, t) = [u[2], 0.0]
    function bc!(res, u, p, t)
        res[1] = u(pi / 4)[1] + pi / 2
        res[2] = u(pi / 2)[1] - pi / 2
        return nothing
    end
    bc(u, p, t) = [u(pi / 4)[1] + pi / 2, u(pi / 2)[1] - pi / 2]
    coupled(u, p, t) = [u(0.17)[1] + u(1.63)[1] - 3.6, u(0.31)[2] - 2]
    for alg in (Ascher1(), Ascher3(), Ascher5()), adaptive in (false, true)
        for prob in (
                BVProblem(f!, bc!, zeros(2), (0.0, 2.0)),
                BVProblem(f, bc, zeros(2), (0.0, 2.0)),
            )
            sol = solve(prob, alg; dt = 0.2, adaptive, abstol = 1.0e-8)
            @test successful_retcode(sol)
            @test sol(0.73) ≈ [4 * 0.73 - 3pi / 2, 4] atol = 1.0e-7
            @test maximum(abs, bc(sol, nothing, sol.t)) < 1.0e-7
        end
        sol = solve(
            BVProblem(f, coupled, zeros(2), (0.0, 2.0)), alg;
            dt = 0.2, adaptive, abstol = 1.0e-8
        )
        @test successful_retcode(sol)
        @test sol(0.73) ≈ [1.46, 2] atol = 1.0e-7

        # Exercise both interpolation call signatures, including vector times
        # returned as a DiffEqArray and in-place calls through the solution API.
        ts = [0.17, 0.73, 1.63]
        values = sol(ts)
        @test values.t == ts
        @test Array(values) ≈ [2 .* ts'; fill(2.0, 1, length(ts))] atol = 1.0e-7
        @test Array(sol(ts; idxs = 1)) ≈ 2 .* ts atol = 1.0e-7
        @test Array(sol(ts; idxs = [2, 1])) ≈ [fill(2.0, 1, length(ts)); 2 .* ts'] atol = 1.0e-7
        for continuity in (:left, :right)
            out = zeros(2)
            sol(out, 0.73; continuity)
            @test out ≈ [1.46, 2] atol = 1.0e-7
            selected = zeros(1)
            sol(selected, 0.73; idxs = [1], continuity)
            @test selected ≈ [1.46] atol = 1.0e-7
        end
    end
end

@testset "Ascher boundary interpolation for DAE components" begin
    using LinearAlgebra: Diagonal
    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = u[2] - t^2
        return nothing
    end
    function bc!(res, u, p, t)
        res[1] = u(0.37)[1] + u(0.71)[2] - (0.37^3 / 3 + 0.71^2 + 2)
        return nothing
    end
    fun = BVPFunction(
        f!, bc!; mass_matrix = Diagonal([1.0, 0.0]),
        bcresid_prototype = zeros(1)
    )
    prob = BVProblem(fun, zeros(2), (0.0, 1.0))
    for adaptive in (false, true)
        sol = solve(prob, Ascher3(); dt = 0.2, adaptive, abstol = 1.0e-8)
        @test successful_retcode(sol)
        @test sol(0.37) ≈ [2 + 0.37^3 / 3, 0.37^2] atol = 1.0e-7
        res = zeros(1)
        bc!(res, sol, nothing, sol.t)
        @test abs(res[1]) < 1.0e-7
    end
end

@testset "Adaptive interpolation after mesh redistribution" begin
    f!(du, u, p, t) = (du[1] = -p * u[1]; nothing)
    f(u, p, t) = -p .* u
    bc!(res, u, p, t) = (res[1] = u(0.037)[1] - exp(-p * 0.037); nothing)
    bc(u, p, t) = [u(0.037)[1] - exp(-p * 0.037)]
    for prob in (
            BVProblem(f!, bc!, [1.0], (0.0, 1.0), 20.0),
            BVProblem(f, bc, [1.0], (0.0, 1.0), 20.0),
        )
        sol = solve(prob, Ascher3(); dt = 0.1, adaptive = true, abstol = 1.0e-6)
        @test successful_retcode(sol)
        @test maximum(abs(sol(t)[1] - exp(-20t)) for t in range(0, 1; length = 21)) < 1.0e-6
        @test maximum(abs(u[1] - exp(-20t)) for (u, t) in zip(sol.u, sol.t)) < 1.0e-6
        @test abs(first(bc(sol, 20.0, sol.t))) < 1.0e-8
    end
end
