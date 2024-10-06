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
    function bc1!(res, u, p, t)
        res[1] = u[1]
        res[2] = u[3] - 1
        res[3] = u[2] - sin(1.0)
    end
    function f1_analytic(u, p, t)
        return [sin(t), sin(t), 1.0, 0.0]
    end
    u01 = [0.0, 0.0, 0.0, 0.0]
    tspan1 = (0.0, 1.0)
    zeta1 = [0.0, 0.0, 1.0]
    fun1 = BVPFunction(f1!, bc1!, analytic = f1_analytic,
        mass_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0])
    prob1 = BVProblem(fun1, u01, tspan1)
    SOLVERS = [alg(zeta = zeta1)
               for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)]
    for stage in (1, 2, 3, 4, 5, 6, 7)
        sol = solve(prob1, SOLVERS[stage], dt = 0.01)
        SciMLBase.successful_retcode(sol)
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

    function bc2!(res, u, p, t)
        res[1] = u[1] + 1
        res[2] = u[2] + 2
        res[3] = u[3] - 1
    end
    u02 = [0.0, 0.0, 0.0, 0.0, 0.0]
    tspan2 = (0.0, 1.0)
    zeta2 = [0.0, 1.0, 1.0]
    fun2 = BVPFunction(
        f2!, bc2!, mass_matrix = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0])
    prob2 = BVProblem(fun2, u02, tspan2)
    SOLVERS = [alg(zeta = zeta2)
               for alg in (Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7)]
    for stage in (1, 2, 3, 4, 5, 6, 7)
        sol = solve(prob2, SOLVERS[stage], dt = 0.01)
        SciMLBase.successful_retcode(sol)
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

#TODO: Add more test cases
