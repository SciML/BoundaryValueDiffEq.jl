using BoundaryValueDiffEqFIRK, Test, SparseArrays, ForwardDiff
const F = BoundaryValueDiffEqFIRK

regression_f!(du, u, p, t) = (du[1] = u[2]; du[2] = u[1]; nothing)
regression_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = u(t[end])[1] - exp(t[end]); nothing)
regression_nonlinear!(du, u, p, t) = (du[1] = u[2]; du[2] = 2 * u[1]^3; nothing)
regression_nonlinear_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 0.5; r[2] = u(t[end])[1] - 1; nothing)
function regression_matrix!(du, u, p, t)
    for j in 1:2
        du[1, j] = u[2, j]
        du[2, j] = u[1, j]
    end
    return nothing
end
function regression_matrix_bc!(r, u, p, t)
    a, b = u(t[1]), u(t[end])
    r[1] = a[1] - 1
    r[2] = b[1] - exp(t[end])
    r[3] = a[3] - 2
    r[4] = b[3] - 2exp(t[end])
    return nothing
end
function regression_nlls_bc!(r, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = u(t[end])[1] - exp(t[end])
    r[3] = u(t[1])[2] - 1
    return nothing
end

function regression_packed(prob, alg; kwargs...)
    return F.__init_firk_device(prob, alg, F.__device_initial_state(prob.u0, prob.p, first(prob.tspan)); kwargs...)
end
function test_device_regressions(upload, platform)
    @testset "Nonlinear adaptive BVP" begin
        prob = BVProblem(regression_nonlinear!, regression_nonlinear_bc!, upload([0.7, 0.5]), (0.0, 1.0))
        cache = regression_packed(prob, RadauIIa3(; platform); dt = 0.5, abstol = 1.0e-6)
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.4)) ≈ [1 / 1.6, 1 / 1.6^2] atol = 1.0e-5
        @test length(sol.t) > 3
    end
    @testset "Float32 matrix states" begin
        prob = BVProblem(regression_matrix!, regression_matrix_bc!, upload(ones(Float32, 2, 2)), (0.0f0, 1.0f0))
        sol = solve!(regression_packed(prob, RadauIIa3(; platform); dt = 0.1f0, adaptive = false, abstol = 1.0f-5))
        @test successful_retcode(sol.retcode)
        @test size(sol(0.3f0)) == (2, 2)
        @test eltype(sol.u[1]) == Float32
        @test Array(sol(0.3f0)) ≈ Float32[exp(0.3) 2exp(0.3); exp(0.3) 2exp(0.3)] atol = 2.0f-4
    end
    @testset "Sparse least squares" begin
        fun = BVPFunction(regression_f!, regression_nlls_bc!; bcresid_prototype = upload(zeros(3)))
        prob = BVProblem(fun, upload([0.5, 0.5]), (0.0, 1.0); nlls = Val(true))
        sol = solve!(regression_packed(prob, RadauIIa3(; platform); dt = 0.1, adaptive = false))
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.4)) ≈ fill(exp(0.4), 2) atol = 1.0e-5
    end
    return @testset "Function and mesh initial guesses" begin
        initial = (p, t) -> upload([exp(t), exp(t)])
        prob = BVProblem(regression_f!, regression_bc!, initial, (0.0, 1.0))
        sol = solve!(regression_packed(prob, RadauIIa3(; platform); dt = 0.2, adaptive = false))
        @test successful_retcode(sol.retcode)
        guess = [initial(nothing, t) for t in (0.0, 0.1, 0.4, 1.0)]
        prob2 = remake(prob; u0 = guess)
        sol2 = solve!(regression_packed(prob2, RadauIIa3(; platform); adaptive = false))
        @test successful_retcode(sol2.retcode)
        @test Array(sol2(0.4)) ≈ fill(exp(0.4), 2) atol = 1.0e-4
    end
end

function test_host_offload(platform)
    @testset "Host-owned FIRK: nested=$nested" for nested in (false, true)
        prob = BVProblem(regression_f!, regression_bc!, [0.5, 0.5], (0.0, 1.0))
        alg = RadauIIa3(;
            platform, nested_nlsolve = nested,
            jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff()))
        )
        cache = init(prob, alg; dt = 0.2, adaptive = false)
        if platform isa CPU && !nested
            device = F.__firk_offload_cache_impl(platform, prob, cache.alg, prob.u0, cache.TU)
            fields = map(fieldnames(typeof(cache))) do name
                name === :device_cache ? device : getfield(cache, name)
            end
            cache = F.FIRKCacheExpand{true, Float64, F.DiffCacheNeeded, false}(fields...)
        end
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test sol(0.4) ≈ fill(exp(0.4), 2) atol = 1.0e-4
        @test sol.u[1] isa Vector
    end
    return @testset "Offloaded nested residual and AD" begin
        prob = BVProblem(regression_f!, regression_bc!, [0.5, 0.5], (0.0, 1.0))
        cache = init(prob, RadauIIa3(nested_nlsolve = true); dt = 0.2, adaptive = false)
        device = F.__firk_offload_cache_impl(platform, prob, cache.alg, prob.u0, cache.TU)
        K = reshape(collect(1.0:6), 2, 3)
        p = [0.2, 0.1, 0.5, 0.7]
        reference = similar(K)
        F.FIRK_nlsolve!(reference, K, p, prob.f.f, cache.TU, prob.p, prob.f.mass_matrix)
        result = similar(K)
        F.__firk_offload_nested!(result, K, p, prob.f.f, cache.TU, prob.p, device, Val(true))
        @test result ≈ reference
        device_jac = ForwardDiff.jacobian(vec(K)) do x
            out = similar(x)
            F.__firk_offload_nested!(
                reshape(out, size(K)), reshape(x, size(K)), p,
                prob.f.f, cache.TU, prob.p, device, Val(true)
            )
            out
        end
        host_jac = ForwardDiff.jacobian(vec(K)) do x
            out = similar(x)
            F.FIRK_nlsolve!(
                reshape(out, size(K)), reshape(x, size(K)), p,
                prob.f.f, cache.TU, prob.p, prob.f.mass_matrix
            )
            out
        end
        @test device_jac ≈ host_jac
    end
end

if !isdefined(@__MODULE__, :FIRK_GPU_TESTS)
    test_device_regressions(identity, CPU())
    test_host_offload(CPU())
end

function regression_offload_cache(cache, platform)
    device = F.__firk_offload_cache_impl(
        platform, cache.prob, cache.alg,
        zeros(eltype(cache), cache.M), cache.TU
    )
    fields = map(fieldnames(typeof(cache))) do name
        name === :device_cache ? device : getfield(cache, name)
    end
    Constructor = Core.apply_type(F.FIRKCacheExpand, typeof(cache).parameters[1:4]...)
    return Constructor(fields...)
end
regression_parameter_f!(du, u, p, t) = (du[1] = u[2]; du[2] = p[1] * u[1]; nothing)
regression_parameter_a!(r, u, p) = (r[1] = u[1] - 1; r[2] = u[2] - 1; nothing)
regression_parameter_b!(r, u, p) = (r[1] = u[1] - exp(1); nothing)
regression_constraint_f!(du, u, p, t) = (du[1] = u[2]; du[2] = u[3]; nothing)
regression_constraint_bc!(r, u, p, t) = (r[1] = u(t[1])[1]; r[2] = u(t[end])[1]; r[3] = u(t[end])[2]; nothing)
function test_offload_features(platform)
    @testset "Offloaded parameter fitting" begin
        prob = TwoPointBVProblem(
            regression_parameter_f!, (regression_parameter_a!, regression_parameter_b!),
            [1.0, 1.0], (0.0, 1.0), [0.8]; bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true
        )
        cache = init(prob, RadauIIa3(; platform); dt = 0.1, adaptive = false)
        platform isa CPU && (cache = regression_offload_cache(cache, platform))
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test sol.prob.p ≈ [1.0] atol = 1.0e-5
    end
    return @testset "Offloaded rectangular constraints" begin
        fun = BVPFunction(regression_constraint_f!, regression_constraint_bc!; f_prototype = zeros(2))
        prob = BVProblem(fun, [0.1, 0.2, 0.3], (0.0, 1.0); lb = fill(-Inf, 3))
        cache = init(prob, RadauIIa3(); dt = 0.2, adaptive = false)
        u = vec(copy(cache.y₀))
        expected = [zeros(2) for _ in 1:(length(cache.y) - 1)]
        result = deepcopy(expected)
        F.Φ!(expected, cache, cache.y, u, F.DiffCacheNeeded(), Val(true))
        offload = regression_offload_cache(cache, platform)
        F.Φ!(result, offload, offload.y, u, F.DiffCacheNeeded(), Val(true))
        @test reduce(vcat, result) ≈ reduce(vcat, expected)
    end
end
if !isdefined(@__MODULE__, :FIRK_GPU_TESTS)
    test_offload_features(CPU())
end

regression_singular_f!(du, u, p, t) = (du[1] = u[2]; du[2] = 0; nothing)
regression_derivative_bc!(r, u, p, t) = (r[1] = u[1, 1] - 1; r[2] = u.du[end][1] - 4; nothing)
regression_singular_fit_f!(du, u, p, t) = (du[1] = u[2]; du[2] = p[1]; nothing)
regression_singular_fit_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = u(t[1])[2] - 2; r[3] = u(t[end])[1] - 4; nothing)
function test_device_singular(upload, platform)
    return @testset "Singular term and derivative boundary condition" begin
        prob = BVProblem(
            regression_singular_f!, regression_derivative_bc!, upload([1.0, 2.0]),
            (1.0, 2.0); singular_term = upload([0.0 0.0; 0.0 1.0])
        )
        sol = solve!(regression_packed(prob, RadauIIa3(; platform); dt = 0.5, abstol = 1.0e-7))
        @test successful_retcode(sol.retcode)
        @test Array(sol(1.4)) ≈ [1.4^2, 2.8] atol = 1.0e-6
        @test Array(sol(1.4, Val{1})) ≈ [2.8, 2.0] atol = 1.0e-6
        fit = BVProblem(
            BVPFunction(regression_singular_fit_f!, regression_singular_fit_bc!; bcresid_prototype = zeros(3)),
            upload([1.0, 2.0]), (1.0, 2.0), upload([0.2]);
            singular_term = upload([0.0 0.0; 0.0 1.0]), tune_parameters = true
        )
        fitted = solve!(regression_packed(fit, RadauIIa3(; platform); dt = 0.5, abstol = 1.0e-7))
        @test successful_retcode(fitted.retcode)
        @test Array(fitted.prob.p) ≈ [0.0] atol = 1.0e-6
        @test Array(fitted(1.4)) ≈ [1.4^2, 2.8] atol = 1.0e-6
    end
end
if !isdefined(@__MODULE__, :FIRK_GPU_TESTS)
    test_device_singular(identity, CPU())
end
