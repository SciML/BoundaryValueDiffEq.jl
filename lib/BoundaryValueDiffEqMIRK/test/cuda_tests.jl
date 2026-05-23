@testitem "GPU accelerated MIRK tests" begin
    using CUDA, BoundaryValueDiffEqMIRK, CUDSS
    @inline function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end
    @inline function f(u, p, t)
        return SVector{2}(u[2], -u[1])
    end

    @inline function bc_a!(ra, ua, p)
        ra[1] = ua[1] - 5
        return nothing
    end
    @inline function bc_a(ua, p)
        return [ua[1] - 5]
    end

    @inline function bc_b!(rb, ub, p)
        rb[1] = ub[1] - 0
        return nothing
    end
    @inline function bc_b(ub, p)
        return [ub[1] - 0]
    end
    @inline function bc!(res, sol, p, t)
        res[1] = sol(0.0)[1] - 5.0
        res[2] = sol(5.0)[1] - 0.0
    end
    @inline function bc(sol, p, t)
        return [sol(0.0)[1] - 5.0, sol(5.0)[1] - 0.0]
    end
    u0 = CuArray([0.0, 0.0])
    tspan = (0.0f0, 5.0f0)
    prob1_iip = TwoPointBVProblem(f!, (bc_a!, bc_b!), u0, tspan, bcresid_prototype = (zeros(Float32, 1), zeros(Float32, 1)))
    prob1_oop = TwoPointBVProblem(f, (bc_a, bc_b), u0, tspan, bcresid_prototype = (zeros(Float32, 1), zeros(Float32, 1)))
    prob2_iip = BVProblem(f!, bc!, u0, tspan)
    prob2_oop = BVProblem(f, bc, u0, tspan)

    sol1_iip = solve(prob1_iip, MIRK4(); dt = 0.01f0, adaptive = false)
    @test_nowarn SciMLBase.successful_retcode(sol1_iip)
    sol1_oop = solve(prob1_oop, MIRK4(); dt = 0.01f0, adaptive = false)
    @test_nowarn SciMLBase.successful_retcode(sol1_oop)
    sol2_iip = solve(prob2_iip, MIRK4(); dt = 0.01f0, adaptive = false)
    @test_nowarn SciMLBase.successful_retcode(sol2_iip)
    sol2_oop = solve(prob2_oop, MIRK4(); dt = 0.01f0, adaptive = false)
    @test_nowarn SciMLBase.successful_retcode(sol2_oop)
end
