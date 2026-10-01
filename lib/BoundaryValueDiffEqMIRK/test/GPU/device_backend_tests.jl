using BoundaryValueDiffEqMIRK
using ADTypes: AutoFiniteDiff, AutoSparse
using BoundaryValueDiffEqCore: BVPJacobianAlgorithm
using SciMLBase: BVPFunction, BVProblem, ODEFunction, TwoPointBVProblem, solve, successful_retcode
using StaticArrays: SVector
using Test

# GPU RHS callbacks return static values to avoid device allocation.
function exponential!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[1]
    return nothing
end
exponential(u, p, t) = SVector(u[2], u[1])

function exponential_bc!(res, u, p, t)
    res[1] = u(0.0)[1] - 1
    res[2] = u(1.0)[1] - exp(1.0)
    return nothing
end
exponential_bc(u, p, t) = [u(0.0)[1] - 1, u(1.0)[1] - exp(1.0)]
exponential_bca!(res, u, p) = (res[1] = u[1] - 1)
exponential_bcb!(res, u, p) = (res[1] = u[1] - exp(1.0))
exponential_bca(u, p) = [u[1] - 1]
exponential_bcb(u, p) = [u[1] - exp(1.0)]

function exponential_problems()
    u0, tspan = [1.0, 1.0], (0.0, 1.0)
    bcresid_prototype = (zeros(1), zeros(1))
    return (
        BVProblem(exponential!, exponential_bc!, u0, tspan),
        BVProblem(exponential, exponential_bc, u0, tspan),
        TwoPointBVProblem(
            exponential!, (exponential_bca!, exponential_bcb!), u0, tspan;
            bcresid_prototype
        ),
        TwoPointBVProblem(
            exponential, (exponential_bca, exponential_bcb), u0, tspan;
            bcresid_prototype
        ),
    )
end

parameter_rate(p::Number) = p
parameter_rate(p::Union{Tuple, AbstractArray}) = p[1]
parameter_rate(p::NamedTuple) = p.rate[1]
function parameterized!(du, u, p, t)
    du[1] = u[2]
    du[2] = parameter_rate(p) * u[1]
    return nothing
end

function exponential_matrix!(du, u, p, t)
    for j in 1:2
        du[1, j] = u[2, j]
        du[2, j] = u[1, j]
    end
    return nothing
end
function exponential_matrix_bc!(res, u, p, t)
    res[1] = u(0.0)[1] - 1
    res[2] = u(1.0)[1] - exp(1.0)
    res[3] = u(0.0)[3] - 2
    res[4] = u(1.0)[3] - 2 * exp(1.0)
    return nothing
end

function adaptive_exponential!(du, u, p, t)
    du[1] = u[2]
    du[2] = 16 * u[1]
    return nothing
end
function adaptive_bc!(res, u, p, t)
    # Off-mesh BCs also require the stages produced on the GPU on the host.
    res[1] = u(0.13)[1] - exp(4 * (0.13 - 1))
    res[2] = u(0.87)[1] - exp(4 * (0.87 - 1))
    return nothing
end

function eigenvalue!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * u[1]
    return nothing
end
function eigenvalue_bca!(res, u, p)
    res[1] = u[1]
    res[2] = u[2] - pi
    return nothing
end
eigenvalue_bcb!(res, u, p) = (res[1] = u[1])

function test_device_backend(platform; algorithms = (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6, MIRK6I))
    @testset "$Alg / problem $i" for Alg in algorithms,
            (i, prob) in enumerate(exponential_problems())
        # The default sparse Jacobian exercises tracer fallback on the host
        # followed by numerical ForwardDiff Dual evaluation on the device.
        cpu = solve(prob, Alg(); dt = 0.1, adaptive = false, abstol = 1.0e-10)
        gpu = solve(prob, Alg(; platform); dt = 0.1, adaptive = false, abstol = 1.0e-10)
        @test successful_retcode(cpu)
        @test successful_retcode(gpu)
        @test Array(gpu) ≈ Array(cpu) rtol = 1.0e-8 atol = 1.0e-9
        @test gpu(0.37) ≈ cpu(0.37) rtol = 1.0e-8 atol = 1.0e-9
        @test gpu(0.37, Val{1}) ≈ cpu(0.37, Val{1}) rtol = 1.0e-8 atol = 1.0e-8
        @test all(isapprox(y, fill(exp(t), 2); rtol = 5.0e-3) for (y, t) in zip(gpu.u, gpu.t))
        @test gpu.u[1] isa Vector{Float64}
    end

    @testset "Finite differences" for prob in exponential_problems()
        jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoFiniteDiff()))
        sol = solve(prob, MIRK4(; platform, jac_alg); dt = 0.1, adaptive = false)
        @test successful_retcode(sol)
        @test sol.u[5] ≈ fill(exp(sol.t[5]), 2) rtol = 1.0e-5
    end

    @testset "ODEFunction with host metadata" begin
        ode = ODEFunction(exponential!; jac_prototype = zeros(2, 2))
        # Keep the ODE-specific Jacobian prototype on the nested ODEFunction.
        bvp_f = BVPFunction(ode, exponential_bc!; jac_prototype = nothing)
        prob = BVProblem(bvp_f, [1.0, 1.0], (0.0, 1.0))
        @test !isbits(ode)
        @test BoundaryValueDiffEqMIRK.__device_function(prob.f.f) === exponential!
        cpu = solve(prob, MIRK4(); dt = 0.1, adaptive = false)
        gpu = solve(prob, MIRK4(; platform); dt = 0.1, adaptive = false)
        @test successful_retcode(cpu)
        @test successful_retcode(gpu)
        @test Array(gpu) ≈ Array(cpu) rtol = 1.0e-8 atol = 1.0e-9
        @test gpu(0.37) ≈ cpu(0.37) rtol = 1.0e-8 atol = 1.0e-9
    end

    @testset "Parameter transfer" for p in (1.0, (1.0,), [1.0], (; rate = [1.0]))
        prob = BVProblem(parameterized!, exponential_bc!, [1.0, 1.0], (0.0, 1.0), p)
        sol = solve(prob, MIRK4(; platform); dt = 0.1, adaptive = false)
        @test successful_retcode(sol)
        @test sol(0.37) ≈ fill(exp(0.37), 2) rtol = 1.0e-5
    end

    @testset "Matrix state" begin
        prob = BVProblem(
            exponential_matrix!, exponential_matrix_bc!, [1.0 2.0; 1.0 2.0], (0.0, 1.0)
        )
        cpu = solve(prob, MIRK4(); dt = 0.1, adaptive = false)
        sol = solve(prob, MIRK4(; platform); dt = 0.1, adaptive = false)
        @test successful_retcode(cpu)
        @test successful_retcode(sol)
        @test size(sol(0.37)) == size(cpu(0.37))
        @test sol(0.37) ≈ cpu(0.37) rtol = 1.0e-8 atol = 1.0e-9
        @test vec(sol(0.37)) ≈ exp(0.37) .* [1, 1, 2, 2] rtol = 1.0e-5
    end

    @testset "Adaptive mesh and dense interpolation" begin
        prob = BVProblem(adaptive_exponential!, adaptive_bc!, [0.5, 1.0], (0.0, 1.0))
        sol = solve(prob, MIRK4(; platform); dt = 0.25, abstol = 1.0e-7)
        @test successful_retcode(sol)
        @test length(sol.t) > 5
        @test sol(0.43) ≈ [exp(4 * (0.43 - 1)), 4 * exp(4 * (0.43 - 1))] rtol = 1.0e-5
        @test sol(0.43, Val{1}) ≈ [4 * exp(4 * (0.43 - 1)), 16 * exp(4 * (0.43 - 1))] rtol = 1.0e-4
    end

    @testset "Singular term" begin
        # A regular tspan isolates transfer/evaluation of the singular matrix
        # from regularity constraints at the singular endpoint.
        function singular_bc!(res, u, p, t)
            res[1] = u(1.0)[1] - 1
            res[2] = u(2.0)[1] - 2
            return nothing
        end
        prob = BVProblem(
            exponential!, singular_bc!, [1.0, 1.0], (1.0, 2.0);
            singular_term = [0.0 0.0; 0.0 -2.0]
        )
        cpu = solve(prob, MIRK4(); dt = 0.05, adaptive = false)
        gpu = solve(prob, MIRK4(; platform); dt = 0.05, adaptive = false)
        @test successful_retcode(cpu)
        @test successful_retcode(gpu)
        @test Array(gpu) ≈ Array(cpu) rtol = 1.0e-8 atol = 1.0e-9
    end

    @testset "Tuned vector parameters" begin
        guess(p, t) = [sin(pi * t), pi * cos(pi * t)]
        prob = TwoPointBVProblem(
            eigenvalue!, (eigenvalue_bca!, eigenvalue_bcb!), guess, (0.0, 1.0), [9.0];
            bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true
        )
        sol = solve(prob, MIRK4(; platform); dt = 0.05, adaptive = false)
        @test successful_retcode(sol)
        @test sol.prob.p[1] ≈ pi^2 rtol = 1.0e-5
        @test all(
            isapprox(y, [sin(pi * t), pi * cos(pi * t)]; rtol = 1.0e-4, atol = 1.0e-5)
                for (y, t) in zip(sol.u, sol.t)
        )
    end
    return nothing
end
