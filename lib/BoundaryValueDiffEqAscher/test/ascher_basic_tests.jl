# Standard test BVDAE problem from the URI M. ASCHER and RAYMOND J. SPITERI paper
@testitem "Test Ascher solver on example problem 1" begin
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
        return [(1 + u[2] - sin(t)) * u[4] + cos(t), cos(t),
            u[4], (u[1] - sin(t)) * (u[4] - e^t)]
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
        f1!, analytic = f1_analytic, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
    fun_oop = ODEFunction(
        f1, analytic = f1_analytic, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
    prob_iip = BVProblem(fun_iip, bc1!, u01, tspan1)
    prob_oop = BVProblem(fun_oop, bc1, u01, tspan1)
    tpprob_iip = TwoPointBVProblem(
        fun_iip, (bca1!, bcb1!), u01, tspan1, bcresid_prototype = (zeros(2), zeros(1)))
    tpprob_oop = TwoPointBVProblem(
        fun_oop, (bca1, bcb1), u01, tspan1, bcresid_prototype = (zeros(2), zeros(1)))
    prob1Arr = [prob_iip, prob_oop, tpprob_iip, tpprob_oop]
    SOLVERS = [alg(zeta = zeta1)
               for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)]
    for i in 1:4
        for stage in (3, 4, 5, 6, 7)
            sol = solve(prob1Arr[i], SOLVERS[stage], dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
            @test sol.errors[:final] < 1e-4
        end
    end
end

### Another BVDAE problem ###
# Comes from "Boundary value problems for differential-algebraic equations"
# by Leonid V. Kalachev and Robert E. O'Malley
@testitem "Test Ascher solver on example problem 2" begin
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
            u[2] + u[3] + u[5] + 1, u[2] + u[4], u[1] + u[5], u[1] + u[2] + 1, u[1] + u[3]]
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
        f2!, bc2!, mass_matrix = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0])
    fun2_oop = BVPFunction(
        f2, bc2, mass_matrix = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0])
    prob2_iip = BVProblem(fun2_iip, u02, tspan2)
    prob2_oop = BVProblem(fun2_oop, u02, tspan2)
    prob2Arr = [prob2_iip, prob2_oop]
    SOLVERS = [alg(zeta = zeta2)
               for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)]
    for i in 1:2
        for stage in (2, 4, 5, 6)
            sol = solve(prob2Arr[i], SOLVERS[stage], dt = 0.01, adaptive = false)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

@testitem "Test Ascher solver on example problem 3" begin
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
    SOLVERS = [alg(zeta = zeta3)
               for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)]
    for i in 1:2
        for stage in (2, 3, 4, 5, 6, 7)
            sol = solve(prob3Arr[i], SOLVERS[stage], dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

@testitem "JET tests" begin
    using JET, BoundaryValueDiffEqAscher, SciMLBase
    function f1!(du, u, p, t)
        e = 2.7
        du[1] = (1 + u[2] - sin(t)) * u[4] + cos(t)
        du[2] = cos(t)
        du[3] = u[4]
        du[4] = (u[1] - sin(t)) * (u[4] - e^t)
    end

    function bc1!(res, u, p, t)
        res[1] = u[1]
        res[2] = u[3] - 1
        res[3] = u[2] - sin(1.0)
    end
    u01 = [0.0, 0.0, 0.0, 0.0]
    tspan1 = (0.0, 1.0)
    zeta1 = [0.0, 0.0, 1.0]
    fun1 = BVPFunction(f1!, bc1!, mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
    prob1 = BVProblem(fun1, u01, tspan1)
    SOLVERS = [alg(zeta = zeta1)
               for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)]
    for stage in (1, 2, 3, 4, 5, 6, 7)
        @test_call target_modules=(BoundaryValueDiffEqAscher,) solve(
            prob1, SOLVERS[stage], dt = 0.01)
    end
    #@test_opt target_modules=(BoundaryValueDiffEqAscher,) solve(prob1, Ascher4(zeta = zeta1), dt = 0.01)
end
