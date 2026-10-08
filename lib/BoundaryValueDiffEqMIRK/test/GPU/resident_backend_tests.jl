using BoundaryValueDiffEqMIRK: BoundaryValueDiffEqCore
using BoundaryValueDiffEqMIRK
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using SciMLBase: BVPFunction, BVProblem, ReturnCode, TwoPointBVProblem, init, solve, solve!, successful_retcode
using Test

include("resizing_tests.jl")

function resident_exponential!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * u[1]
    return nothing
end
resident_exponential(u, p, t) = (u[2], p[1] * u[1])

function resident_bc!(res, u, p, t)
    res[1] = u(t[1])[1] - 1
    res[2] = u(t[end])[1] - exp(t[end])
    return nothing
end
resident_bc(u, p, t) = (u(t[1])[1] - 1, u(t[end])[1] - exp(t[end]))
function resident_bca!(res, u, p)
    res[1] = u[1] - 1
    return nothing
end
function resident_bcb!(res, u, p)
    res[1] = u[1] - exp(one(eltype(u)))
    return nothing
end
resident_bca(u, p) = (u[1] - 1,)
resident_bcb(u, p) = (u[1] - exp(one(eltype(u))),)

function resident_problem(to_device, ::Type{T}, iip, twopoint) where {T}
    u0 = to_device(T[0.5, 0.5])
    p = to_device(T[1])
    tspan = (zero(T), one(T))
    f = iip ? resident_exponential! : resident_exponential
    if twopoint
        bc = iip ? (resident_bca!, resident_bcb!) : (resident_bca, resident_bcb)
        bcresid_prototype = (to_device(zeros(T, 1)), to_device(zeros(T, 1)))
        return TwoPointBVProblem(f, bc, u0, tspan, p; bcresid_prototype, nlls = Val(false))
    end
    bc = iip ? resident_bc! : resident_bc
    return BVProblem(f, bc, u0, tspan, p)
end

function check_resident_storage(cache, is_device, is_jacobian_device = is_device)
    @test is_device(BoundaryValueDiffEqMIRK.__mirk_states(cache))
    @test is_device(cache.residual)
    @test is_jacobian_device(BoundaryValueDiffEqMIRK.__mirk_jacobian(cache))
    @test size(BoundaryValueDiffEqMIRK.__mirk_jacobian(cache), 2) == length(BoundaryValueDiffEqMIRK.__mirk_states(cache))
    @test size(BoundaryValueDiffEqMIRK.__mirk_jacobian(cache), 1) == length(cache.residual)
    return nothing
end

function resident_matrix!(du, u, p, t)
    for j in 1:2
        du[1, j] = u[2, j]
        du[2, j] = u[1, j]
    end
    return nothing
end
function resident_matrix_bc!(res, u, p, t)
    ua, ub = u(t[1]), u(t[end])
    res[1] = ua[1] - 1
    res[2] = ub[1] - exp(t[end])
    res[3] = ua[3] - 2
    res[4] = ub[3] - 2 * exp(t[end])
    return nothing
end

function resident_interior_bc!(res, u, p, t)
    ta, tb = oftype(t[1], 0.13), oftype(t[1], 0.87)
    res[1] = u(ta)[1] - exp(4 * (ta - 1))
    res[2] = u(tb)[1] - exp(4 * (tb - 1))
    return nothing
end

function resident_nonlinear!(du, u, p, t)
    du[1] = u[2]
    du[2] = 2 * u[1]^3
    return nothing
end
function resident_nonlinear_bc!(res, u, p, t)
    res[1] = u(t[1])[1] - 0.5
    res[2] = u(t[end])[1] - 1
    return nothing
end

function resident_nlls_bc!(res, u, p, t)
    res[1] = u(t[1])[1] - 1
    res[2] = u(t[end])[1] - exp(t[end])
    res[3] = u(t[1])[2] - 1
    return nothing
end

function test_resident_backend(
        to_device, is_device, platform;
        algorithms = (MIRK2, MIRK3, MIRK4, MIRK5, MIRK6, MIRK6I),
        element_types = (Float32, Float64), nlsolve = nothing, nllssolve = nothing,
        is_jacobian_device = is_device
    )
    @testset "$Alg / $T / inplace=$iip / twopoint=$twopoint" for Alg in algorithms,
            T in element_types, iip in (true, false), twopoint in (false, true)
        prob = resident_problem(to_device, T, iip, twopoint)
        guess = Array(prob.u0)
        abstol = T === Float32 ? T(1.0e-5) : T(1.0e-9)
        cache = init(prob, Alg(; platform, nlsolve); dt = T(0.1), adaptive = false, abstol)
        check_resident_storage(cache, is_device, is_jacobian_device)
        @test eltype(BoundaryValueDiffEqMIRK.__mirk_states(cache)) === T
        @test eltype(cache.residual) === T
        @test eltype(BoundaryValueDiffEqMIRK.__mirk_jacobian(cache)) === T
        sol = solve!(cache)
        @test successful_retcode(sol)
        check_resident_storage(cache, is_device, is_jacobian_device)
        @test all(is_device, sol.u)
        @test is_device(sol.original.u)
        @test all(u -> eltype(u) === T, sol.u)
        @test Array(prob.u0) == guess
        @test all(
            isapprox(Array(u), fill(exp(t), 2); rtol = 5.0e-3, atol = 5.0e-4)
                for (u, t) in zip(sol.u, sol.t)
        )
        interpolated = sol(T(0.37))
        @test is_device(interpolated)
        @test Array(interpolated) ≈ fill(exp(T(0.37)), 2) rtol = 5.0e-3 atol = 5.0e-4
    end

    @testset "Resident matrix state" begin
        prob = BVProblem(
            resident_matrix!, resident_matrix_bc!, to_device([1.0 2.0; 1.0 2.0]),
            (0.0, 1.0)
        )
        sol = solve(prob, MIRK4(; platform, nlsolve); dt = 0.1, adaptive = false)
        @test successful_retcode(sol)
        @test all(is_device, sol.u)
        @test size(sol.u[1]) == (2, 2)
        interpolated = sol(0.37)
        @test is_device(interpolated)
        @test size(interpolated) == (2, 2)
        @test Array(interpolated) ≈ exp(0.37) .* [1.0 2.0; 1.0 2.0] rtol = 1.0e-5
    end

    @testset "Resident nonlinear Newton iteration" begin
        prob = BVProblem(
            resident_nonlinear!, resident_nonlinear_bc!, to_device([0.75, 0.5]), (0.0, 1.0)
        )
        sol = solve(prob, MIRK4(; platform, nlsolve); dt = 0.1, adaptive = false, abstol = 1.0e-9)
        @test successful_retcode(sol)
        @test is_device(sol.original.u)
        @test all(is_device, sol.u)
        @test all(
            isapprox(Array(u), [1 / (2 - t), 1 / (2 - t)^2]; rtol = 1.0e-4)
                for (u, t) in zip(sol.u, sol.t)
        )
    end

    @testset "Resident Jacobian algorithms" begin
        prob = resident_problem(to_device, Float64, true, false)
        for jac_alg in (
                BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 1)),
                BVPJacobianAlgorithm(
                    AutoForwardDiff(;
                        chunksize = 2,
                        tag = BoundaryValueDiffEqMIRK.ForwardDiff.Tag(resident_exponential!, Float64)
                    )
                ),
                BVPJacobianAlgorithm(AutoFiniteDiff()),
                BVPJacobianAlgorithm(AutoFiniteDiff(; fdjtype = Val(:central))),
                BVPJacobianAlgorithm(;
                    bc_diffmode = AutoFiniteDiff(),
                    nonbc_diffmode = AutoForwardDiff(; chunksize = 2)
                ),
            )
            sol = solve(prob, MIRK4(; platform, nlsolve, jac_alg); dt = 0.2, adaptive = false)
            @test successful_retcode(sol)
            @test is_device(sol.original.u)
            @test Array(sol.u[end]) ≈ fill(exp(1.0), 2) rtol = 1.0e-4
        end
    end

    @testset "Resident nonlinear least squares" begin
        bf = BVPFunction(
            resident_exponential!, resident_nlls_bc!;
            bcresid_prototype = to_device(zeros(3))
        )
        prob = BVProblem(
            bf, to_device([0.5, 0.5]), (0.0, 1.0), to_device([1.0]); nlls = Val(true)
        )
        sol = solve(
            prob, MIRK4(; platform, nlsolve = nllssolve);
            dt = 0.1, adaptive = false, abstol = 1.0e-8
        )
        @test successful_retcode(sol)
        @test is_device(sol.original.u)
        @test all(is_device, sol.u)
        @test Array(sol.u[end]) ≈ fill(exp(1.0), 2) rtol = 1.0e-4
    end

    @testset "Device initial-guess forms and backend inference" begin
        p = to_device([1.0, 1.0])
        guesses = (
            to_device([0.5, 0.5]),
            view(to_device([0.5, 99.0, 0.5]), 1:2:3),
            [to_device(fill(exp(t), 2)) for t in range(0.0, 1.0; length = 6)],
            (p, t) -> p .* exp(t),
        )
        for guess in guesses
            prob = BVProblem(
                resident_exponential!, resident_bc!, guess, (0.0, 1.0), p
            )
            # The storage backend is inferred from the device initial guess.
            sol = solve(prob, MIRK4(; nlsolve); dt = 0.2, adaptive = false)
            @test successful_retcode(sol)
            @test all(is_device, sol.u)
            @test Array(sol.u[end]) ≈ fill(exp(1.0), 2) rtol = 1.0e-4
        end
    end

    @testset "Resident adaptive mesh and interior boundary conditions" begin
        prob = BVProblem(
            resident_exponential!, resident_interior_bc!, to_device([0.5, 1.0]),
            (0.0, 1.0), to_device([16.0])
        )
        # The defect divides algebraic residuals by the mesh width. Solve the
        # nonlinear system tightly enough to measure interpolation/refinement.
        cache = init(
            prob, MIRK4(; platform, nlsolve); dt = 0.25, abstol = 1.0e-7,
            nlsolve_kwargs = (; abstol = 1.0e-11)
        )
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test length(sol.t) > 5
        @test length(sol.t) <= 129
        check_resident_storage(cache, is_device, is_jacobian_device)
        @test all(is_device, sol.u)
        interpolated = sol(0.43)
        @test is_device(interpolated)
        @test Array(interpolated) ≈ [exp(4 * (0.43 - 1)), 4 * exp(4 * (0.43 - 1))] rtol = 1.0e-4
        derivative = sol(0.43, Val{1})
        @test is_device(derivative)
        @test Array(derivative) ≈ [4 * exp(4 * (0.43 - 1)), 16 * exp(4 * (0.43 - 1))] rtol = 1.0e-4
    end

    @testset "Resident solve termination" begin
        prob = BVProblem(
            resident_exponential!, resident_interior_bc!, to_device([0.5, 1.0]),
            (0.0, 1.0), to_device([16.0])
        )

        @testset "No error control on a fixed mesh" begin
            cache = init(
                prob, MIRK4(; platform, nlsolve, max_num_subintervals = 4);
                dt = 0.25, adaptive = false, controller = NoErrorControl(), abstol = 1.0e-7
            )
            original_mesh = copy(cache.host_mesh)
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test sol.t == original_mesh
            @test cache.host_mesh == original_mesh
            @test all(is_device, sol.u)
        end

        @testset "Refinement stops at the interval limit" begin
            cache = init(
                prob, MIRK4(; platform, nlsolve, max_num_subintervals = 4);
                dt = 0.25, abstol = 1.0e-7
            )
            original_mesh = copy(cache.host_mesh)
            sol = solve!(cache)
            @test successful_retcode(sol.original)
            @test sol.retcode == ReturnCode.Failure
            @test sol.t == original_mesh
            @test cache.host_mesh == original_mesh
            @test all(is_device, sol.u)
        end

        @testset "Nonlinear failure / adaptive=$adaptive" for adaptive in (false, true)
            cache = init(
                resident_problem(to_device, Float64, true, false), MIRK4(; platform, nlsolve);
                dt = 0.25, adaptive, nlsolve_kwargs = (; maxiters = 0)
            )
            original_mesh = copy(cache.host_mesh)
            original_guess = Array(BoundaryValueDiffEqMIRK.__mirk_states(cache))
            sol = solve!(cache)
            @test !successful_retcode(sol.original)
            @test sol.retcode == sol.original.retcode
            @test sol.t == original_mesh
            @test cache.host_mesh == original_mesh
            @test Array(BoundaryValueDiffEqMIRK.__mirk_states(cache)) == original_guess
            @test all(is_device, sol.u)
        end
    end

    @testset "Flat resident buffer lifecycle" begin
        for Alg in (MIRK2, MIRK4, MIRK6I)
            cache = init(
                resident_problem(to_device, Float64, true, false),
                Alg(; platform, nlsolve); dt = 0.25, adaptive = false, abstol = 1.0e-11
            )
            test_resident_resizing(cache, is_device)
        end
    end

    @testset "Immutable resident cache refinement and reuse" begin
        MIRK = BoundaryValueDiffEqMIRK
        prob = BVProblem(
            resident_exponential!, resident_interior_bc!, to_device([0.5, 1.0]),
            (0.0, 1.0), to_device([16.0])
        )
        jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))
        # Keep nonlinear residuals below the mesh-scaled defect tolerance so
        # this test exercises buffer resizing and reuse on a refined mesh.
        cache = init(
            prob, MIRK4(; platform, nlsolve, jac_alg); dt = 0.25, abstol = 1.0e-7,
            nlsolve_kwargs = (; abstol = 1.0e-11)
        )
        initial_cache, host_mesh = cache, cache.host_mesh
        fields = (
            :mesh, :mesh_dt, :y, :k_discrete, :k_interp, :collocation_cache,
            :fᵢ₂_cache, :residual, :errors,
        )
        buffers = map(name -> getproperty(cache, name), fields)
        @test cache isa MIRK.MIRKCache
        @test !ismutabletype(typeof(cache))
        @testset "Concrete device storage: $name" for (name, reference) in zip(fields, buffers)
            @test reference isa AbstractVector
            @test isconcretetype(typeof(reference)) && is_device(reference)
        end

        # Warm a Dual workspace on the coarse mesh, so refinement has stale
        # dimensions to replace rather than allocating the first workspace late.
        BoundaryValueDiffEqMIRK.__device_jacobian!(BoundaryValueDiffEqMIRK.__mirk_jacobian(cache), vec(BoundaryValueDiffEqMIRK.__mirk_states(cache)), cache)
        dual_type = only(T for T in keys(cache.device_cache) if T <: MIRK.ForwardDiff.Dual)
        coarse_work = cache.device_cache[dual_type]
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test length(sol.t) > 5
        @test cache === initial_cache
        @test cache.host_mesh === host_mesh
        @test all(getproperty(cache, name) === reference for (name, reference) in zip(fields, buffers))
        @test cache.host_mesh == sol.t == Array(cache.mesh)
        @test length(cache.mesh_dt) == length(sol.t) - 1
        @test size(BoundaryValueDiffEqMIRK.__mirk_states(cache), 2) == length(sol.t)
        @test all(
            size(array, ndims(array)) == length(sol.t) - 1
                for array in (
                    MIRK.__mirk_stages(cache), MIRK.__mirk_interp_stages(cache),
                    MIRK.__mirk_collocation(cache), MIRK.__mirk_rhs_tmp(cache),
                )
        )
        @test length(cache.errors) == length(sol.t) - 1
        check_resident_storage(cache, is_device, is_jacobian_device)
        refined_work = cache.device_cache[dual_type]
        @test all(new !== old for (new, old) in zip(values(refined_work), values(coarse_work)))
        @test map(size, values(refined_work)) == map(size, values(cache.device_cache[Float64]))

        # Compare Jacobians on the refined mesh, including the interior BCs.
        ad_jacobian = copy(BoundaryValueDiffEqMIRK.__mirk_jacobian(cache))
        BoundaryValueDiffEqMIRK.__device_jacobian!(ad_jacobian, vec(BoundaryValueDiffEqMIRK.__mirk_states(cache)), cache)
        fd_jacobian = similar(BoundaryValueDiffEqMIRK.__mirk_states(cache), eltype(ad_jacobian), size(ad_jacobian))
        BoundaryValueDiffEqCore.__bvp_device_ad_jacobian!(
            fd_jacobian, vec(BoundaryValueDiffEqMIRK.__mirk_states(cache)), cache,
            AutoFiniteDiff(; fdjtype = Val(:central)), axes(fd_jacobian, 1), true
        )
        @test collect(ad_jacobian) ≈ Array(fd_jacobian) rtol = 2.0e-6 atol = 2.0e-7

        times = (0.13, 0.43, 0.87)
        saved_mesh = copy(sol.t)
        saved_values = map(t -> Array(sol(t)), times)
        saved_derivatives = map(t -> Array(sol(t, Val{1})), times)
        # Overwrite current working arrays before replacing them by bisection:
        # the saved solution must own its state, stages and mesh metadata.
        fill!(BoundaryValueDiffEqMIRK.__mirk_states(cache), 0)
        fill!(BoundaryValueDiffEqMIRK.__mirk_stages(cache), 0)
        fill!(BoundaryValueDiffEqMIRK.__mirk_interp_stages(cache), 0)
        @test map(t -> Array(sol(t)), times) == saved_values
        @test map(t -> Array(sol(t, Val{1})), times) == saved_derivatives
        MIRK.half_mesh!(cache)
        @test isempty(cache.device_cache)
        @test sol.t == saved_mesh
        repeated = solve!(cache)
        @test successful_retcode(repeated)
        @test length(repeated.t) >= 2 * (length(saved_mesh) - 1) + 1
        @test all(getproperty(cache, name) === reference for (name, reference) in zip(fields, buffers))
        @test cache.host_mesh === host_mesh
        @test map(size, values(cache.device_cache[dual_type])) ==
            map(size, values(cache.device_cache[Float64]))
        @test sol.t == saved_mesh
        @test map(t -> Array(sol(t)), times) == saved_values
        @test map(t -> Array(sol(t, Val{1})), times) == saved_derivatives
    end
    return nothing
end
