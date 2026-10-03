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
        res[1] = u[1]
        res[2] = u[3] - 1
        res[3] = u[2] - sin(1.0)
    end
    function bc1(u, p, t)
        return [u[1], u[3] - 1, u[2] - sin(1.0)]
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
    zeta1 = [0.0, 0.0, 1.0]
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
        alg(zeta = zeta1)
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
        res[1] = u[1] + 1
        res[2] = u[2] + 2
        res[3] = u[3] - 1
    end
    function bc2(u, p, t)
        return [u[1] + 1, u[2] + 2, u[3] - 1]
    end
    u02 = [0.0, 0.0, 0.0, 0.0, 0.0]
    tspan2 = (0.0, 1.0)
    zeta2 = [0.0, 1.0, 1.0]
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
        alg(zeta = zeta2)
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
        res[1] = u[1]
        res[2] = u[2]
    end
    function bc3(u, p, t)
        return [u[1], u[2]]
    end
    function f3_analytic(u, p, t)
        return [sin(t - 1), sin(t - 1), -cos(t - 1)]
    end
    u03 = [0.0, 0.0, 0.0]
    tspan3 = (0.0, 1.0)
    zeta3 = [1.0, 1.0]
    fun_iip = ODEFunction(f3!, analytic = f3_analytic, mass_matrix = [1 0 0; 0 1 0; 0 0 0])
    fun_oop = ODEFunction(f3, analytic = f3_analytic, mass_matrix = [1 0 0; 0 1 0; 0 0 0])
    prob_iip = BVProblem(fun_iip, bc3!, u03, tspan3)
    prob_oop = BVProblem(fun_oop, bc3, u03, tspan3)
    prob3Arr = [prob_iip, prob_oop]
    SOLVERS = [
        alg(zeta = zeta3)
            for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)
    ]
    for i in 1:2
        for stage in (2, 3, 4, 5, 6, 7)
            sol = solve(prob3Arr[i], SOLVERS[stage], dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

@testset "Ascher initial guess from u0" begin
    using BoundaryValueDiffEqAscher, SciMLBase
    using ForwardDiff: value

    # `f` and `bc` are undefined at the zero state the iterate used to start from
    function f_recip!(du, u, p, t)
        iszero(value(u[1])) && error("f! was evaluated at u[1] = 0")
        du[1] = 1 / u[1] - 1
    end
    function bc_recip!(res, u, p, t)
        iszero(value(u[1])) && error("bc! was evaluated at u[1] = 0")
        res[1] = 1 / u[1] - 1
    end
    prob = BVProblem(
        BVPFunction(f_recip!, bc_recip!; mass_matrix = ones(1, 1)), [1.0], (0.0, 1.0)
    )
    @testset "f and bc undefined at zero, adaptive = $adaptive" for adaptive in (false, true)
        sol = solve(prob, Ascher4(zeta = [1.0]); dt = 0.01, adaptive)
        @test SciMLBase.successful_retcode(sol)
        @test all(x -> isapprox(only(x), 1.0; atol = 1.0e-4), sol.u)
    end

    # algebraic component undefined at zero
    function f_dae!(du, u, p, t)
        iszero(value(u[2])) && error("f! was evaluated at u[2] = 0")
        du[1] = 1 / u[2] + u[1] - 2
        du[2] = u[2] - 1
    end
    bc_dae!(res, u, p, t) = (res[1] = u[1] - 1)
    prob = BVProblem(
        BVPFunction(f_dae!, bc_dae!; mass_matrix = [1.0 0.0; 0.0 0.0]), [1.0, 1.0],
        (0.0, 1.0)
    )
    @testset "algebraic guess, adaptive = $adaptive" for adaptive in (false, true)
        sol = solve(prob, Ascher4(zeta = [1.0]); dt = 0.05, adaptive)
        @test SciMLBase.successful_retcode(sol)
        @test all(u -> isapprox(u, [1.0, 1.0]; atol = 1.0e-6), sol.u)
    end

    # non-constant solution u(t) = sqrt(1 + 2t)
    f_sqrt!(du, u, p, t) = (du[1] = 1 / u[1])
    bc_sqrt!(res, u, p, t) = (res[1] = u[1] - sqrt(3.0))
    prob = BVProblem(
        BVPFunction(f_sqrt!, bc_sqrt!; mass_matrix = ones(1, 1)), [1.5], (0.0, 1.0)
    )
    @testset "non-constant solution, adaptive = $adaptive" for adaptive in (false, true)
        sol = solve(prob, Ascher4(zeta = [1.0]); dt = 0.05, adaptive)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs.(first.(sol.u) .- sqrt.(1 .+ 2 .* sol.t))) < 1.0e-6
    end

    # sampled and functional guesses, including one with fewer samples than mesh points
    f3!(du, u, p, t) = (du[1] = -u[3]; du[2] = -u[3]; du[3] = u[2] - sin(t - 1))
    bc3!(res, u, p, t) = (res[1] = u[1]; res[2] = u[2])
    sol3(t) = [sin(t - 1), sin(t - 1), -cos(t - 1)]
    guesses = (
        "101 samples" => [sol3(t) for t in range(0, 1; length = 101)],
        "11 samples" => [sol3(t) for t in range(0, 1; length = 11)],
        "function" => (p, t) -> sol3(t) .+ 0.2,
    )
    @testset "$name guess, adaptive = $adaptive" for (name, u0) in guesses,
            adaptive in (false, true)
        prob = BVProblem(
            BVPFunction(f3!, bc3!; mass_matrix = [1.0 0 0; 0 1 0; 0 0 0]), u0, (0.0, 1.0)
        )
        sol = solve(prob, Ascher4(zeta = [1.0, 1.0]); dt = 0.01, adaptive)
        @test SciMLBase.successful_retcode(sol)
        @test maximum(maximum(abs.(sol.u[i] .- sol3(sol.t[i]))) for i in eachindex(sol.t)) <
            1.0e-6
    end
end

# JET tests have been moved to the separate QA test group (test/qa/)
