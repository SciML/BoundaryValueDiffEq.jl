using BoundaryValueDiffEqMIRK
using SciMLBase: BVProblem, init
using Test

const MIRK = BoundaryValueDiffEqMIRK

function interpolation_test_f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p * u[1] + t
    return nothing
end
function interpolation_test_bc!(res, u, p, t)
    res[1] = u(first(t))[1]
    res[2] = u(last(t))[1]
    return nothing
end
interpolation_test_f(u, p, t) = [u[2], -p * u[1] + t]
interpolation_test_bc(u, p, t) = [u(first(t))[1], u(last(t))[1]]

@testset "Device interpolation on the CPU backend" begin
    # Compare device kernels with the existing CPU interpolation path.
    @testset "$Alg / $T / $AD / iip=$iip" for Alg in (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6, MIRK6I),
            T in (Float32, Float64), AD in (AutoFiniteDiff, AutoForwardDiff), iip in (true, false)
        alg = Alg(; jac_alg = BVPJacobianAlgorithm(AD()))
        f = iip ? interpolation_test_f! : interpolation_test_f
        bc = iip ? interpolation_test_bc! : interpolation_test_bc
        prob = BVProblem(
            f, bc, T[0.3, 0.7],
            (zero(T), one(T)), T(1.2)
        )
        cache = init(prob, alg; dt = T(0.25), adaptive = false)
        u = copy(cache.y₀_flat)
        u .+= range(zero(T), T(0.4); length = length(u))
        trait = MIRK.__cache_trait(cache.alg.jac_alg)
        y = MIRK.recursive_unflatten!(cache.y, u)
        if iip
            residual = [zeros(T, 2) for _ in cache.mesh_dt]
            MIRK.Φ!(residual, cache, y, u, trait, Val(false))
        else
            MIRK.Φ(cache, y, u, trait)
        end
        MIRK.interp_setup!(cache)
        states = [copy(MIRK._collocation_tmp(v, u, trait)) for v in y]
        for (dest, src) in zip(cache.y₀.u, states)
            copyto!(dest, src)
        end
        cpu_interp = MIRK.MIRKInterpolation(cache.mesh, states, cache)
        cpu_eval = MIRK.EvalSol(states, cache.mesh, cache)
        packed_y = reduce(hcat, states)
        K = cat(
            [copy(MIRK._collocation_tmp(k, u, trait)) for k in cache.k_discrete]...;
            dims = 3
        )
        KI = zeros(T, 2, cache.ITU.s_star - cache.stage, length(cache.mesh_dt))
        tmp = zeros(T, 2, length(cache.mesh_dt))
        algid = Val(nameof(Alg))
        MIRK.__mirk_device_interp_setup!(
            CPU(), KI, tmp, K, packed_y, f, prob.p,
            cache.mesh, cache.mesh_dt, cache.ITU, (2,), Val(iip), nothing
        )
        tol = T === Float32 ? 2.0f-4 : 1.0e-12
        @test KI ≈ cat(cache.k_interp.u...; dims = 3) rtol = tol atol = tol
        evalsol = MIRK.EvalSol(
            MIRK.__build_interpolation(packed_y, K, KI, cache.mesh, cache.mesh_dt, algid, (2,))
        )
        interp = MIRK.__build_interpolation(
            packed_y, K, KI, cache.mesh, cache.mesh_dt, algid, (2,), CPU()
        )
        for t in T.((0.13, 0.51, 0.87))
            expected = cpu_interp(t, nothing, Val{0}, prob.p)
            derivative = cpu_interp(t, nothing, Val{1}, prob.p)
            @test collect(evalsol(t)) ≈ expected rtol = tol atol = tol
            @test collect(evalsol(t, Val{1})) ≈ derivative rtol = tol atol = tol
            @test interp(t, nothing, Val{0}, prob.p) ≈ expected rtol = tol atol = tol
            @test interp(t, 1, Val{0}, prob.p) ≈ expected[1] rtol = tol atol = tol
        end
        @test collect(evalsol(zero(T))) == cpu_eval(zero(T))
        @test collect(evalsol(one(T))) == cpu_eval(one(T))
        @test evalsol[1] == states[1]
        @test evalsol[:, end] == states[end]
        @test evalsol[1, 2] == states[2][1]
        @test evalsol.u[end] == states[end]
        @test collect(evalsol.du[2]) ≈ cpu_interp(cache.mesh[2], nothing, Val{1}, prob.p) rtol = tol atol = tol

        new_mesh = collect(zero(T):T(0.125):one(T))
        new_y = zeros(T, 2, length(new_mesh))
        MIRK.__mirk_device_refine!(
            CPU(), new_y, new_mesh, packed_y, K, KI, cache.mesh,
            cache.mesh_dt, algid, (2,)
        )
        expected_refined = reduce(hcat, [cpu_eval(t) for t in new_mesh])
        @test new_y ≈ expected_refined rtol = tol atol = tol

        errors = zeros(T, length(cache.mesh_dt))
        defect = MIRK.__mirk_device_defect!(
            CPU(), errors, tmp, similar(tmp), K, KI, packed_y,
            f, prob.p, cache.mesh, cache.mesh_dt,
            algid, (2,), Val(iip), nothing, cache.ITU.τ_star
        )
        # The CPU estimator uses the same two defect sample points, but evaluates
        # the stages through its existing matrix products and VectorOfArray cache.
        cpu_errors = MIRK.VectorOfArray([zeros(T, 2) for _ in cache.mesh_dt])
        cpu_defect, _ = MIRK.error_estimate!(
            cache, DefectControl(), cpu_errors, nothing, nothing, T(1.0e-6)
        )
        expected_errors = [maximum(abs, e) for e in cpu_errors.u]
        @test errors ≈ expected_errors rtol = tol atol = tol
        @test defect ≈ cpu_defect rtol = tol atol = tol
    end
end

@testset "Device solution indexing and output selection" begin
    mesh, mesh_dt = [0.0, 1.0], [1.0]
    y = reshape(collect(1.0:8.0), 4, 2)
    K, KI = fill(4.0, 4, 3, 1), fill(4.0, 4, 1, 1)
    sol = MIRK.EvalSol(MIRK.__build_interpolation(y, K, KI, mesh, mesh_dt, Val(:MIRK4), (2, 2)))
    interp = MIRK.__build_interpolation(
        y, K, KI, mesh, mesh_dt, Val(:MIRK4), (2, 2), CPU()
    )
    @test size(sol) == (2, 2, 2)
    @test length(sol.u) == length(sol.du) == 2
    @test collect(first(sol.du)) ≈ fill(4.0, 2, 2)
    @test collect(last(sol.du)) ≈ fill(4.0, 2, 2)
    @test sol[2, 1, 2] == y[2, 2]
    @test sol[1, 2, 2] == y[3, 2]
    @test sol[:, :, 2] == reshape(y[:, 2], 2, 2)
    @test sol.u[2][2, 1] == y[2, 2]
    @test sol(0.25)[2, 1] ≈ 3.0
    @test_throws MethodError sol(0.25, Val{2})
    @test_throws ArgumentError interp(0.25, nothing, Val{2}, nothing)
    expected = reshape(y[:, 1] .+ 1.0, 2, 2)
    @test interp(0.25, nothing, Val{0}, nothing) ≈ expected
    @test interp(0.25, nothing, Val{0}, Val(:parameter), :left) ≈ expected
    @test interp(0.25, [4, 1], Val{0}, nothing) ≈ expected[[4, 1]]
    @test interp(0.25, 2, Val{1}, nothing) ≈ 4.0
    @test_throws BoundsError interp(0.25, 5, Val{0}, nothing)
    @test_throws BoundsError interp(0.25, [0, 1], Val{0}, nothing)
    @test_throws DimensionMismatch interp(zeros(1), 0.25, nothing, Val{0}, nothing)
    @test_throws DimensionMismatch interp(zeros(2), 0.25, 1, Val{0}, nothing)
    out = similar(expected)
    interp(out, 0.25, nothing, Val{0}, nothing)
    @test out ≈ expected
    fill!(out, 0.0)
    interp(out, 0.25, nothing, Val{0}, :parameter)
    @test out ≈ expected
    times = [0.25, 0.75]
    @test Array(interp(times, 1, Val{0}, nothing)) ≈ [2.0, 4.0]
    @test Array(interp(times, 1, Val{0}, Val(:parameter), :left)) ≈ [2.0, 4.0]
    @test Array(interp(times, [4, 1], Val{0}, nothing)) ≈ [5.0 7.0; 2.0 4.0]

    tensor_y = reshape(collect(1.0:16.0), 8, 2)
    tensor_sol = MIRK.EvalSol(
        MIRK.__build_interpolation(
            tensor_y, fill(8.0, 8, 3, 1), fill(8.0, 8, 1, 1),
            mesh, mesh_dt, Val(:MIRK4), (2, 2, 2)
        )
    )
    @test tensor_sol[2, 1, 2, 2] == tensor_y[6, 2]
end

# Slow CPU arrays and their views must retain the CPU backend.
struct MIRKSlowCPUVector <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(u::MIRKSlowCPUVector) = size(u.data)
Base.IndexStyle(::Type{MIRKSlowCPUVector}) = IndexLinear()
Base.getindex(u::MIRKSlowCPUVector, i::Int) = u.data[i]
MIRK.fast_scalar_indexing(::Type{MIRKSlowCPUVector}) = false

@testset "CPU initial backend detection" begin
    using StaticArrays: SVector
    array = [1.0, 2.0]
    static_array = SVector(1.0, 2.0)
    slow_array = MIRKSlowCPUVector(array)
    @test !MIRK.fast_scalar_indexing(slow_array)
    for u in (
            array, static_array, view(array, 1:2), view(static_array, 1:2),
            slow_array, view(slow_array, 1:2),
        )
        @test MIRK.__device_initial_backend(u) isa CPU
    end
end
