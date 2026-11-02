using BoundaryValueDiffEqFIRK, Test, LinearAlgebra, SparseArrays
const NDF = BoundaryValueDiffEqFIRK

nested_rhs!(du, u, p, t) = (du[1] = u[2]; du[2] = p[1] * u[1]; nothing)
nested_rhs(u, p, t) = (u[2], p[1] * u[1])
nested_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = u(t[end])[1] - exp(t[end]); nothing)
nested_bca(u, p) = (u[1] - 1,)
nested_bcb(u, p) = (u[1] - exp(1),)
nested_nonlinear!(du, u, p, t) = (du[1] = u[1]^2; nothing)
nested_interior!(r, u, p, t) = (r[1] = u(0.23)[1] - 1 / 1.77; nothing)
nested_matrix!(du, u, p, t) = (du[1, 1] = u[1, 2]; du[1, 2] = u[1, 1]; nothing)
nested_matrix_bc!(r, u, p, t) = (r[1] = u(t[1])[1, 1] - 1; r[2] = u(t[end])[1, 1] - exp(t[end]); nothing)
nested_fit_bc!(r, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = u(t[1])[2] - 1; r[3] = u(t[end])[1] - exp(t[end]); nothing)
nested_mass!(du, u, p, t) = (du[1] = 2u[2]; du[2] = 3u[1]; nothing)
nested_dae!(du, u, p, t) = (du[1] = u[2]; du[2] = u[2] - cos(t); nothing)
nested_dae_bc!(r, u, p, t) = (r[1] = u(t[1])[1]; r[2] = u(t[1])[2] - 1; nothing)

nested_init(prob, alg; kwargs...) = NDF.__init_firk_device(
    prob, alg, NDF.__device_initial_state(prob.u0, prob.p, first(prob.tspan)); kwargs...
)
function nested_problem(upload, twopoint)
    u0, p = upload([0.5, 0.5]), upload([1.0])
    return twopoint ? TwoPointBVProblem(
            nested_rhs, (nested_bca, nested_bcb), u0, (0.0, 1.0), p;
            bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
        ) : BVProblem(nested_rhs!, nested_bc!, u0, (0.0, 1.0), p)
end

function test_nested_device(upload, platform)
    @testset "Resident nested $Alg two-point=$twopoint" for Alg in (
                RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7,
                LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5,
                LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5,
                LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5,
            ), twopoint in (false, true)
        prob = nested_problem(upload, twopoint)
        alg = Alg(; platform, nested_nlsolve = true)
        cache = platform isa CPU ? nested_init(prob, alg; dt = 0.05, adaptive = false) :
            init(prob, alg; dt = 0.05, adaptive = false)
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) atol = (Alg == RadauIIa1 ? 0.06 : 0.003)
        @test length(BoundaryValueDiffEqFIRK.__firk_unknowns(cache)) == 2length(sol.t)
        @test size(BoundaryValueDiffEqFIRK.__firk_jacobian(cache)) == (length(BoundaryValueDiffEqFIRK.__firk_unknowns(cache)), length(BoundaryValueDiffEqFIRK.__firk_unknowns(cache)))
        @test size(BoundaryValueDiffEqFIRK.__firk_states(cache), 2) == (length(sol.t) - 1) * (NDF.alg_stage(alg) + 1) + 1
        work = NDF.__firk_nested_buffers(cache, Float64)
        @test all(iszero, work.status)
        @test typeof(work.stages) == typeof(BoundaryValueDiffEqFIRK.__firk_unknowns(cache))
        @test typeof(sol.u[1]) == typeof(prob.u0)
    end

    @testset "Implicit stage derivatives, interior BCs and finite differences" begin
        prob = BVProblem(nested_nonlinear!, nested_interior!, upload([0.6]), (0.0, 1.0))
        cache = nested_init(
            prob, RadauIIa3(;
                platform, nested_nlsolve = true, nested_nlsolve_kwargs = (; abstol = 1.0e-13),
                jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(chunksize = 1)))
            ); dt = 0.2, adaptive = false
        )
        u, J = vec(copy(BoundaryValueDiffEqFIRK.__firk_unknowns(cache))), copy(BoundaryValueDiffEqFIRK.__firk_jacobian(cache))
        NDF.__device_jacobian!(J, u, cache)
        reference = zeros(length(cache.residual), length(u))
        x = Array(u)
        for column in eachindex(x)
            plus, minus = copy(x), copy(x)
            plus[column] += 1.0e-5
            minus[column] -= 1.0e-5
            rp, rm = similar(cache.residual), similar(cache.residual)
            NDF.__device_residual!(rp, upload(plus), cache)
            NDF.__device_residual!(rm, upload(minus), cache)
            reference[:, column] .= (Array(rp) .- Array(rm)) ./ 2.0e-5
        end
        @test Matrix(SparseMatrixCSC(J)) ≈ reference atol = 1.0e-7
        # Differentiation must still work after warm starts converge in zero steps.
        NDF.__device_jacobian!(J, u, cache)
        @test Matrix(SparseMatrixCSC(J)) ≈ reference atol = 1.0e-7
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.6)) ≈ [1 / 1.4] atol = 1.0e-5
        for mode in (AutoFiniteDiff(), AutoFiniteDiff(fdjtype = Val(:central)))
            alg = RadauIIa3(;
                platform, nested_nlsolve = true,
                jac_alg = BVPJacobianAlgorithm(AutoSparse(mode))
            )
            sol = solve!(nested_init(prob, alg; dt = 0.2, adaptive = false))
            @test successful_retcode(sol.retcode)
            @test Array(sol(0.6)) ≈ [1 / 1.4] atol = 1.0e-5
        end
    end

    @testset "Nested adaptive controller $controller" for controller in (
            DefectControl(), GlobalErrorControl(), GlobalErrorControl(method = REErrorControl()),
            SequentialErrorControl(), HybridErrorControl(),
        )
        prob = nested_problem(upload, false)
        cache = nested_init(
            prob, RadauIIa3(; platform, nested_nlsolve = true);
            dt = 0.5, abstol = 1.0e-7, controller
        )
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) atol = 1.0e-5
        @test size(BoundaryValueDiffEqFIRK.__firk_jacobian(cache), 2) == length(BoundaryValueDiffEqFIRK.__firk_unknowns(cache))
        @test NDF.__firk_device_error!(cache, controller, 1.0e-7)[2] <= 1.0e-7
        old = Array(sol(0.37))
        fill!(BoundaryValueDiffEqFIRK.__firk_unknowns(cache), 0.7)
        again = solve!(cache)
        @test successful_retcode(again.retcode)
        @test Array(sol(0.37)) == old
    end

    @testset "Nested matrix states, fitting and mass matrices" begin
        prob = BVProblem(nested_matrix!, nested_matrix_bc!, upload(Float32[0.5 0.5]), (0.0, 1.0))
        sol = solve!(
            nested_init(
                prob, LobattoIIIc3(; platform, nested_nlsolve = true);
                dt = 0.1f0, abstol = 1.0e-5, adaptive = false
            )
        )
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.37f0)) ≈ fill(exp(0.37f0), 1, 2) atol = 1.0e-4
        fit = BVProblem(
            BVPFunction(nested_rhs!, nested_fit_bc!; bcresid_prototype = zeros(3)),
            upload([1.0, 1.0]), (0.0, 1.0), upload([0.8]); tune_parameters = true
        )
        sol = solve!(nested_init(fit, RadauIIa3(; platform, nested_nlsolve = true); dt = 0.5, abstol = 1.0e-7))
        @test successful_retcode(sol.retcode)
        @test Array(sol.prob.p) ≈ [1.0] atol = 1.0e-5
        @test length(sol.u[1]) == 2
        mass = BVProblem(
            BVPFunction(nested_mass!, nested_bc!; mass_matrix = upload([2.0 0.0; 0.0 3.0])),
            upload([1.0, 1.0]), (0.0, 1.0)
        )
        sol = solve!(nested_init(mass, RadauIIa3(; platform, nested_nlsolve = true); dt = 0.2, abstol = 1.0e-6))
        @test successful_retcode(sol.retcode)
        @test Array(sol(0.37)) ≈ fill(exp(0.37), 2) atol = 1.0e-5
        dae = BVProblem(
            BVPFunction(nested_dae!, nested_dae_bc!; mass_matrix = upload([1.0 0.0; 0.0 0.0])),
            upload([0.0, 1.0]), (0.0, 1.0)
        )
        for Alg in (RadauIIa3, LobattoIIIc3)
            sol = solve!(nested_init(dae, Alg(; platform, nested_nlsolve = true); dt = 0.05, adaptive = false))
            @test successful_retcode(sol.retcode)
            @test maximum(maximum(abs, Array(sol.u[i]) .- [sin(t), cos(t)]) for (i, t) in enumerate(sol.t)) < 1.0e-6
        end
    end

    test_nested_device_recovery(upload, platform)
    return @testset "Nested failure handling and options" begin
        prob = nested_problem(upload, false)
        cache = nested_init(
            prob, RadauIIa3(;
                platform, nested_nlsolve = true, nested_nlsolve_kwargs = (; maxiters = 0)
            ); dt = 0.2, adaptive = false
        )
        @test !successful_retcode(solve!(cache).retcode)
        @test all(isfinite, BoundaryValueDiffEqFIRK.__firk_states(cache))
        @test_throws ArgumentError nested_init(
            prob, RadauIIa3(;
                platform, nested_nlsolve = true, nested_nlsolve_kwargs = (; maxiters = -1)
            ); dt = 0.2
        )
        @test_throws ArgumentError nested_init(
            prob, RadauIIa3(;
                platform, nested_nlsolve = true, nested_nlsolve_kwargs = (; unsupported = true)
            ); dt = 0.2
        )
        expanded = nested_init(prob, RadauIIa3(; platform); dt = 0.2, adaptive = false)
        nested = nested_init(prob, RadauIIa3(; platform, nested_nlsolve = true); dt = 0.2, adaptive = false)
        @test nnz(BoundaryValueDiffEqFIRK.__firk_jacobian(nested)) < nnz(BoundaryValueDiffEqFIRK.__firk_jacobian(expanded)) ÷ 4
    end
end

function test_nested_device_recovery(upload, platform)
    return @testset "Nested recovery, highest order and least squares" begin
        prob = BVProblem(nested_nonlinear!, nested_interior!, upload([0.6]), (0.0, 1.0))
        cache = nested_init(
            prob, RadauIIa3(;
                platform, nested_nlsolve = true, nested_nlsolve_kwargs = (; maxiters = 2),
                max_num_subintervals = 128
            ); dt = 1.0, abstol = 1.0e-6
        )
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test length(sol.t) > 2
        @test Array(sol(0.6)) ≈ [1 / 1.4] atol = 1.0e-5
        highest = nested_init(
            nested_problem(upload, false), RadauIIa7(; platform, nested_nlsolve = true);
            dt = 0.5, abstol = 1.0e-7, controller = GlobalErrorControl()
        )
        @test successful_retcode(solve!(highest).retcode)
        least_squares = BVProblem(
            BVPFunction(nested_rhs!, nested_fit_bc!; bcresid_prototype = zeros(3)),
            upload([0.5, 0.5]), (0.0, 1.0), upload([1.0])
        )
        ls = solve!(
            nested_init(
                least_squares, RadauIIa3(; platform, nested_nlsolve = true);
                dt = 0.1, adaptive = false
            )
        )
        @test successful_retcode(ls.retcode)
        @test Array(ls(0.37)) ≈ fill(exp(0.37), 2) atol = 1.0e-5
    end
end

if !isdefined(@__MODULE__, :FIRK_GPU_TESTS)
    test_nested_device(identity, CPU())
end
