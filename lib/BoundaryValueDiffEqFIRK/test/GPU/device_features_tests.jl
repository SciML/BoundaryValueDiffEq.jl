using BoundaryValueDiffEqFIRK, Test, SparseArrays, ForwardDiff, LinearAlgebra
using ADTypes: KnownJacobianSparsityDetector
const DF = BoundaryValueDiffEqFIRK

feature_f!(du, u, p, t) = (du[1] = u[2]; du[2] = u[1]; nothing)
feature_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = u(t[end])[1] - exp(t[end]); nothing)
feature_fit_f!(du, u, p, t) = (du[1] = u[2]; du[2] = p[1] * u[1]; nothing)
feature_fit_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = u(t[1])[2] - 1; r[3] = u(t[end])[1] - exp(t[end]); nothing)
feature_fit_a!(r, u, p) = (r[1] = u[1] - 1; r[2] = u[2] - 1; nothing)
feature_fit_b!(r, u, p) = (r[1] = u[1] - exp(1); nothing)
feature_mass_f!(du, u, p, t) = (du[1] = 2u[2]; du[2] = 3u[1]; nothing)
feature_dae_f!(du, u, p, t) = (du[1] = u[2]; du[2] = u[2] - cos(t); nothing)
feature_dae_bc!(r, u, p, t) = (r[1] = u(t[1])[1]; r[2] = u(t[1])[2] - 1; nothing)
feature_timed_bc!(r, u, p, t) = (r[1] = u(p[1])[1] - exp(p[1]); r[2] = u(p[2])[1] - exp(p[2]); nothing)
function feature_fallback_bc!(r, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = (u(t[1])[2] > 0 ? u(t[end])[1] : u(t[end])[2]) - exp(t[end])
    return nothing
end
feature_init(prob, alg; kwargs...) = DF.__init_firk_device(
    prob, alg, DF.__device_initial_state(prob.u0, prob.p, first(prob.tspan)); kwargs...
)

function test_device_features(upload, platform)
    @testset "Resident error controller $(typeof(controller)), $Alg" for
        (Alg, controller) in (
            (RadauIIa3, GlobalErrorControl()),
            (RadauIIa3, GlobalErrorControl(method = REErrorControl())),
            (LobattoIIIa3, SequentialErrorControl()),
            (LobattoIIIb3, HybridErrorControl()),
            (LobattoIIIc3, DefectControl()),
            (RadauIIa7, GlobalErrorControl()),
        )
        prob = BVProblem(feature_f!, feature_bc!, upload([0.5, 0.5]), (0.0, 1.0))
        cache = feature_init(prob, Alg(; platform); dt = 0.5, abstol = 1.0e-6, controller)
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) atol = 1.0e-4
        errors, estimate, info = DF.__firk_device_error!(cache, controller, 1.0e-6)
        @test successful_retcode(info)
        @test estimate <= 1.0e-6
        @test length(errors) == length(sol.t) - 1
    end
    @testset "Resident parameter fitting twopoint=$twopoint" for twopoint in (false, true)
        u0, p = upload([1.0, 1.0]), upload([0.8])
        prob = if twopoint
            TwoPointBVProblem(
                feature_fit_f!, (feature_fit_a!, feature_fit_b!), u0, (0.0, 1.0), p;
                bcresid_prototype = (zeros(2), zeros(1)), tune_parameters = true, nlls = Val(false)
            )
        else
            BVProblem(
                BVPFunction(feature_fit_f!, feature_fit_bc!; bcresid_prototype = zeros(3)),
                u0, (0.0, 1.0), p; tune_parameters = true
            )
        end
        cache = feature_init(prob, RadauIIa3(; platform); dt = 0.5, abstol = 1.0e-7)
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test Array(sol.prob.p) ≈ [1.0] atol = 1.0e-5
        @test length(sol.u[1]) == 2
        @test Array(sol(0.4)) ≈ fill(exp(0.4), 2) atol = 1.0e-5
    end
    @testset "Device mass matrix and DAE" begin
        prob = BVProblem(
            BVPFunction(feature_mass_f!, feature_bc!; mass_matrix = upload([2.0 0.0; 0.0 3.0])),
            upload([1.0, 1.0]), (0.0, 1.0)
        )
        sol = solve!(feature_init(prob, RadauIIa3(; platform); dt = 0.2, abstol = 1.0e-6))
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) atol = 1.0e-5
        dae = BVProblem(
            BVPFunction(feature_dae_f!, feature_dae_bc!; mass_matrix = upload([1.0 0.0; 0.0 0.0])),
            upload([0.0, 1.0]), (0.0, 1.0)
        )
        for Alg in (RadauIIa3, LobattoIIIc3)
            sol = solve!(feature_init(dae, Alg(; platform); dt = 0.05, adaptive = false))
            @test successful_retcode(sol.retcode)
            @test maximum(maximum(abs, Array(sol.u[i]) .- [sin(t), cos(t)]) for (i, t) in enumerate(sol.t)) < 1.0e-6
        end
        @test_throws ArgumentError feature_init(dae, RadauIIa3(; platform); dt = 0.2)
    end
    @testset "Boundary pattern rebuild and conservative fallback" begin
        prob = BVProblem(feature_f!, feature_timed_bc!, upload([0.5, 0.5]), (0.0, 1.0), upload([0.13, 0.87]))
        cache = feature_init(prob, RadauIIa3(; platform); dt = 0.2, adaptive = false)
        sol = solve!(cache)
        old = copy(BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache).pattern)
        copyto!(prob.p, [0.33, 0.67])
        again = solve!(cache)
        @test successful_retcode(again.retcode)
        @test old != BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache).pattern
        @test Array(again(0.5)) ≈ fill(exp(0.5), 2) atol = 1.0e-4
        fallback = BVProblem(feature_f!, feature_fallback_bc!, upload([0.5, 0.5]), (0.0, 1.0))
        fc = feature_init(fallback, RadauIIa3(; platform); dt = 0.2, adaptive = false)
        @test BoundaryValueDiffEqFIRK.__firk_jacobian_plan(fc).boundary_fallback
        @test length(BoundaryValueDiffEqFIRK.__firk_jacobian_plan(fc).groups) == 2
        @test successful_retcode(solve!(fc))
    end
    return @testset "Fixed mesh NoErrorControl and mixed differentiation" begin
        prob = BVProblem(feature_f!, feature_bc!, upload([0.5, 0.5]), (0.0, 1.0))
        alg = LobattoIIIa3(;
            platform, jac_alg = BVPJacobianAlgorithm(
                bc_diffmode = AutoFiniteDiff(fdjtype = Val(:central)),
                nonbc_diffmode = AutoSparse(AutoForwardDiff(chunksize = 2))
            )
        )
        cache = feature_init(prob, alg; dt = 0.5, controller = NoErrorControl())
        @test length(BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache).groups) == 2
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test length(sol.t) == 3
        nested = feature_init(prob, RadauIIa3(; platform, nested_nlsolve = true); dt = 0.1)
        @test length(BoundaryValueDiffEqFIRK.__firk_unknowns(nested)) < length(BoundaryValueDiffEqFIRK.__firk_states(nested))
    end
end

if !isdefined(@__MODULE__, :FIRK_GPU_TESTS)
    test_device_features(identity, CPU())
    @testset "Device interpolation dispatch" begin
        ambiguities = Test.detect_ambiguities(DF; recursive = false)
        @test isempty(filter(pair -> any(m -> occursin("device_", String(m.file)), pair), ambiguities))
        prob = BVProblem(feature_f!, feature_bc!, [0.5, 0.5], (0.0, 1.0))
        sol = solve!(feature_init(prob, RadauIIa3(); dt = 0.2, adaptive = false))
        @test sol.interp(0.4, nothing, Val(1), nothing) ≈ sol(0.4, Val{1})
    end
    @testset "Sparse structure grows linearly with mesh size" begin
        prob = BVProblem(feature_f!, feature_bc!, [0.5, 0.5], (0.0, 1.0))
        for Alg in (RadauIIa3, RadauIIa7, LobattoIIIa5, LobattoIIIb5, LobattoIIIc5)
            small = feature_init(prob, Alg(); dt = 0.1, adaptive = false)
            large = feature_init(prob, Alg(); dt = 0.01, adaptive = false)
            @test nnz(BoundaryValueDiffEqFIRK.__firk_jacobian(large)) < 11nnz(BoundaryValueDiffEqFIRK.__firk_jacobian(small))
            @test maximum(g.ncolors for g in BoundaryValueDiffEqFIRK.__firk_jacobian_plan(large).groups) <= 2 * (DF.alg_stage(Alg()) + 2) * 2
        end
    end
end
