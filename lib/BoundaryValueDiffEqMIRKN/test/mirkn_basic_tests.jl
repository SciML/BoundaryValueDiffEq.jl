@testsetup module MIRKNConvergenceTests

using BoundaryValueDiffEqMIRKN

for order in (4, 6)
    s = Symbol("MIRKN$(order)")
    @eval mirkn_solver(::Val{$order}, args...; kwargs...) = $(s)(args...; kwargs...)
end

function f!(ddu, du, u, p, t)
    ddu[1] = u[1]
end
function f(du, u, p, t)
    return u[1]
end
function bc!(res, du, u, p, t)
    res[1] = u(0.0)[1] - 1
    res[2] = u(1.0)[1]
end
function bc(du, u, p, t)
    return [u(0.0)[1] - 1, u(1.0)[1]]
end
function bc_indexing!(res, du, u, p, t)
    res[1] = u[:, 1][1] - 1
    res[2] = u[:, end][1]
end
function bc_indexing(du, u, p, t)
    return [u[:, 1][1] - 1, u[:, end][1]]
end
function bc_a!(res, du, u, p)
    res[1] = u[1] - 1
end
function bc_b!(res, du, u, p)
    res[1] = u[1]
end
function bc_a(du, u, p)
    return [u[1] - 1]
end
function bc_b(du, u, p)
    return [u[1]]
end
analytical_solution = (u0, p, t) -> [
    (exp(-t) - exp(t - 2)) / (1 - exp(-2)), (-exp(-t) - exp(t - 2)) / (1 - exp(-2))]
u0 = [1.0]
tspan = (0.0, 1.0)
testTol = 0.2
bvpf1 = DynamicalBVPFunction(f!, bc!, analytic = analytical_solution)
bvpf2 = DynamicalBVPFunction(f, bc, analytic = analytical_solution)
bvpf3 = DynamicalBVPFunction(f!, bc_indexing!, analytic = analytical_solution)
bvpf4 = DynamicalBVPFunction(f, bc_indexing, analytic = analytical_solution)
bvpf5 = DynamicalBVPFunction(f!, (bc_a!, bc_b!), analytic = analytical_solution,
    bcresid_prototype = (zeros(1), zeros(1)), twopoint = Val(true))
bvpf6 = DynamicalBVPFunction(f, (bc_a, bc_b), analytic = analytical_solution,
    bcresid_prototype = (zeros(1), zeros(1)), twopoint = Val(true))
probArr = [SecondOrderBVProblem(bvpf1, u0, tspan), SecondOrderBVProblem(bvpf2, u0, tspan),
    SecondOrderBVProblem(bvpf3, u0, tspan), SecondOrderBVProblem(bvpf4, u0, tspan),
    TwoPointSecondOrderBVProblem(bvpf5, u0, tspan),
    TwoPointSecondOrderBVProblem(bvpf6, u0, tspan)]
dts = 1 .// 2 .^ (3:-1:1)

export probArr, dts, testTol, mirkn_solver

end

@testitem "Convergence on Linear" setup=[MIRKNConvergenceTests] begin
    using LinearAlgebra, DiffEqDevTools

    @testset "Problem: $i" for i in (1, 2, 3, 4, 5, 6)
        prob = probArr[i]
        @testset "MIRKN$order" for order in (4, 6)
            sim = test_convergence(
                dts, prob, mirkn_solver(Val(order)); abstol = 1e-8, reltol = 1e-8)
            @test sim.𝒪est[:final]≈order atol=testTol
        end
    end
end

@testitem "JET tests" setup=[MIRKNConvergenceTests] begin
    using JET

    @testset "Problem: $i" for i in 1:6
        prob = probArr[i]
        @testset "MIRKN$order" for order in (4, 6)
            solver = mirkn_solver(Val(order); nlsolve = NewtonRaphson(),
                jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2)))
            @test_call target_modules=(BoundaryValueDiffEqMIRKN,) solve(
                prob, solver; dt = 0.2)
        end
    end
end

@testitem "Example problem from paper" begin
    using BoundaryValueDiffEqMIRKN

    for order in (4, 6)
        s = Symbol("MIRKN$(order)")
        @eval mirkn_solver(::Val{$order}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    function test!(ddu, du, u, p, t)
        ϵ = 0.1
        ddu[1] = u[2]
        ddu[2] = (-u[1] * du[2] - u[3] * du[3]) / ϵ
        ddu[3] = (du[1] * u[3] - u[1] * du[3]) / ϵ
    end
    function test(du, u, p, t)
        ϵ = 0.1
        return [u[2], (-u[1] * du[2] - u[3] * du[3]) / ϵ, (du[1] * u[3] - u[1] * du[3]) / ϵ]
    end

    function bc!(res, du, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(1.0)[1]
        res[3] = u(0.0)[3] + 1
        res[4] = u(1.0)[3] - 1
        res[5] = du(0.0)[1]
        res[6] = du(1.0)[1]
    end

    function bc(du, u, p, t)
        return [u(0.0)[1], u(1.0)[1], u(0.0)[3] + 1, u(1.0)[3] - 1, du(0.0)[1], du(1.0)[1]]
    end

    function bc_indexing!(res, du, u, p, t)
        res[1] = u[:, 1][1]
        res[2] = u[:, end][1]
        res[3] = u[:, 1][3] + 1
        res[4] = u[:, end][3] - 1
        res[5] = du[:, 1][1]
        res[6] = du[:, end][1]
    end

    function bc_indexing(du, u, p, t)
        return [u[:, 1][1], u[:, end][1], u[:, 1][3] + 1,
            u[:, end][3] - 1, du[:, 1][1], du[:, end][1]]
    end

    function bca!(resa, du, u, p)
        resa[1] = u[1]
        resa[2] = u[3] + 1
        resa[3] = du[1]
    end
    function bcb!(resb, du, u, p)
        resb[1] = u[1]
        resb[2] = u[3] - 1
        resb[3] = du[1]
    end

    function bca(du, u, p)
        [u[1], u[3] + 1, du[1]]
    end
    function bcb(du, u, p)
        [u[1], u[3] - 1, du[1]]
    end

    u0 = [1.0, 1.0, 1.0]
    tspan = (0.0, 1.0)

    probArr = [SecondOrderBVProblem(test!, bc!, u0, tspan),
        SecondOrderBVProblem(test, bc, u0, tspan),
        SecondOrderBVProblem(test!, bc_indexing!, u0, tspan),
        SecondOrderBVProblem(test, bc_indexing, u0, tspan),
        TwoPointSecondOrderBVProblem(
            test!, (bca!, bcb!), u0, tspan, bcresid_prototype = (zeros(3), zeros(3))),
        TwoPointSecondOrderBVProblem(
            test, (bca, bcb), u0, tspan, bcresid_prototype = (zeros(3), zeros(3)))]

    @testset "MIRKN$order" for order in (4, 6)
        @testset "Problem $i" for i in 1:6
            sol = solve(probArr[i], mirkn_solver(Val(order)); dt = 0.01)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end
