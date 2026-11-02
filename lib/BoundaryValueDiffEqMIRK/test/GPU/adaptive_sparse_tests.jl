using BoundaryValueDiffEqMIRK: BoundaryValueDiffEqCore
using ADTypes: AutoSparse
using BoundaryValueDiffEqCore: REErrorControl

function cuda_tuning!(du, u, p, t)
    du[1] = p[1] * u[1]
    return nothing
end
function cuda_tuning_bc!(r, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = u(t[end])[1] - exp(2.0)
    return nothing
end

@testset "CUDA parameter tuning with sparse AD and adaptivity" begin
    parameters = CuArray([1.5])
    prob = BVProblem(cuda_tuning!, cuda_tuning_bc!, CuArray([1.0]), (0.0, 1.0), parameters; tune_parameters = true)
    sol = solve(
        prob, MIRK4(; nlsolve = NewtonRaphson(; linsolve = LUFactorization()));
        dt = 0.25, abstol = 1.0e-6, nlsolve_kwargs = (; abstol = 1.0e-11)
    )
    @test successful_retcode(sol)
    @test sol.prob.p isa CuArray
    @test Array(sol.prob.p) ≈ [2.0] rtol = 2.0e-5
    @test Array(parameters) == [1.5]
    @test all(u -> u isa CuArray && length(u) == 1, sol.u)
    @test Array(sol(0.37)) ≈ [exp(0.74)] rtol = 2.0e-5
    @test length(sol.t) > 5
end

# A boundary layer makes uniform refinement waste intervals away from t = 1.
function cuda_layer!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1]^2 * u[1]
    return nothing
end
function cuda_layer_bc!(r, u, p, t)
    r[1] = u(t[1])[1] - exp(-p[1])
    r[2] = u(t[end])[1] - 1
    return nothing
end

@testset "CUDA sparse AD, mesh redistribution and CUDSS" begin
    MIRK = BoundaryValueDiffEqMIRK
    prob = BVProblem(cuda_layer!, cuda_layer_bc!, CuArray([0.1, 0.5]), (0.0, 1.0), CuArray([10.0]))
    for mode in (AutoSparse(AutoForwardDiff(; chunksize = 2)), AutoSparse(AutoFiniteDiff()))
        alg = MIRK4(;
            jac_alg = BVPJacobianAlgorithm(mode),
            nlsolve = NewtonRaphson(; linsolve = LUFactorization())
        )
        cache = init(prob, alg; dt = 0.1, abstol = 1.0e-6, nlsolve_kwargs = (; abstol = 1.0e-11))
        initial_jacobian = MIRK.__mirk_jacobian(cache)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test all(u -> u isa CuArray, sol.u)
        @test sol.original.u isa CuArray
        @test length(sol.t) > 11
        @test maximum(diff(sol.t)) > 1.5minimum(diff(sol.t))
        @test MIRK.__mirk_jacobian(cache) isa cuSPARSE.CuSparseMatrixCSR
        @test size(MIRK.__mirk_jacobian(cache), 2) > size(initial_jacobian, 2)
        @test !MIRK.__mirk_jacobian_plan(cache).boundary_fallback
        @test maximum(g.ncolors for g in MIRK.__mirk_jacobian_plan(cache).groups) <= 8
        @test MIRK.SparseArrays.nnz(MIRK.__mirk_jacobian(cache)) <= 8(length(sol.t) - 1) + 8
        for t in (0.1, 0.6, 0.95)
            @test Array(sol(t)) ≈ [exp(10(t - 1)), 10exp(10(t - 1))] rtol = 2.0e-3 atol = 2.0e-6
        end
        # Verify derivatives after redistribution with independent finite differences.
        reference = similar(MIRK.__mirk_states(cache), size(MIRK.__mirk_jacobian(cache)))
        BoundaryValueDiffEqCore.__bvp_device_ad_jacobian!(
            reference, vec(MIRK.__mirk_states(cache)), cache, AutoFiniteDiff(; fdjtype = Val(:central)),
            axes(reference, 1), true
        )
        MIRK.__device_jacobian!(MIRK.__mirk_jacobian(cache), vec(MIRK.__mirk_states(cache)), cache)
        @test collect(MIRK.__mirk_jacobian(cache)) ≈ Array(reference) rtol = 2.0e-5 atol = 2.0e-6
        @test successful_retcode(solve!(cache))
    end
end

@testset "CUDA error controllers" begin
    prob = resident_problem(CuArray, Float64, true, false)
    for controller in (
            GlobalErrorControl(), GlobalErrorControl(; method = REErrorControl()),
            SequentialErrorControl(), HybridErrorControl(), NoErrorControl(),
        )
        sol = solve(
            prob, MIRK4(; nlsolve = NewtonRaphson(; linsolve = LUFactorization()));
            dt = 0.25, abstol = 1.0e-7, controller, nlsolve_kwargs = (; abstol = 1.0e-12)
        )
        @test successful_retcode(sol)
        @test all(u -> u isa CuArray, sol.u)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) rtol = 5.0e-5
        controller isa NoErrorControl || @test length(sol.t) > 5
    end
    # Exercise the order+2 companion and the Richardson fallback used when
    # no higher-order MIRK companion exists (orders 5 and 6).
    for Alg in (MIRK2, MIRK3, MIRK5, MIRK6, MIRK6I)
        sol = solve(
            prob, Alg(; nlsolve = NewtonRaphson(; linsolve = LUFactorization()));
            dt = 0.25, abstol = 1.0e-7, controller = GlobalErrorControl(),
            nlsolve_kwargs = (; abstol = 1.0e-12)
        )
        @test successful_retcode(sol)
        @test all(u -> u isa CuArray, sol.u)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) rtol = 5.0e-5
    end
end

function cuda_dae!(du, u, p, t)
    du[1] = u[2]
    du[2] = u[2] - cos(t)
    return nothing
end
function cuda_dae_bc!(r, u, p, t)
    r[1] = u(t[1])[1]
    r[2] = u(t[1])[2] - 1
    return nothing
end

@testset "CUDA index-1 DAE" begin
    for mass in ([1.0 0.0; 0.0 0.0], CuArray([1.0 0.0; 0.0 0.0]))
        prob = BVProblem(BVPFunction(cuda_dae!, cuda_dae_bc!; mass_matrix = mass), CuArray([0.0, 1.0]), (0.0, 1.0))
        sol = solve(prob, MIRK4(); dt = 0.01, adaptive = false)
        cpu_prob = BVProblem(
            BVPFunction(cuda_dae!, cuda_dae_bc!; mass_matrix = Array(mass)),
            [0.0, 1.0], (0.0, 1.0)
        )
        cpu = solve(cpu_prob, MIRK4(); dt = 0.01, adaptive = false)
        @test successful_retcode(sol)
        @test successful_retcode(cpu)
        @test reduce(hcat, Array.(sol.u)) ≈ reduce(hcat, cpu.u) rtol = 1.0e-9 atol = 1.0e-10
        @test all(u -> u isa CuArray, sol.u)
        @test maximum(abs(Array(u)[2] - cos(t)) for (u, t) in zip(sol.u, sol.t)) < 1.0e-10
        @test maximum(abs(Array(u)[1] - sin(t)) for (u, t) in zip(sol.u, sol.t)) < 1.0e-5
        @test_throws ArgumentError solve(prob, MIRK4(); dt = 0.05)
    end
end
