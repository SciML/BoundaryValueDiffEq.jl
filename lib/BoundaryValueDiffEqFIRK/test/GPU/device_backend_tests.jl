using BoundaryValueDiffEqFIRK, Test
using ForwardDiff, SparseArrays
using SciMLBase: ReturnCode, isinplace
const FIRK = BoundaryValueDiffEqFIRK

function device_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * u[1]
    return nothing
end
device_rhs(u, p, t) = (u[2], p[1] * u[1])
function device_bc!(r, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = u(t[end])[1] - exp(t[end])
    return nothing
end
device_bc(u, p, t) = (u(t[1])[1] - 1, u(t[end])[1] - exp(t[end]))
device_bca!(r, u, p) = (r[1] = u[1] - 1; nothing)
device_bcb!(r, u, p) = (r[1] = u[1] - exp(one(eltype(u))); nothing)
device_bca(u, p) = (u[1] - 1,)
device_bcb(u, p) = (u[1] - exp(one(eltype(u))),)
function device_interior_bc!(r, u, p, t)
    r[1] = u(oftype(t[1], 0.13))[1] - exp(oftype(t[1], 0.13))
    r[2] = u(oftype(t[1], 0.87))[1] - exp(oftype(t[1], 0.87))
    return nothing
end

function device_problem(upload, T, iip, twopoint; interior = false)
    u0, p = upload(T[0.5, 0.5]), upload(T[1])
    f = iip ? device_rhs! : device_rhs
    if twopoint
        bc = iip ? (device_bca!, device_bcb!) : (device_bca, device_bcb)
        return TwoPointBVProblem(
            f, bc, u0, (zero(T), one(T)), p;
            bcresid_prototype = (upload(zeros(T, 1)), upload(zeros(T, 1))), nlls = Val(false)
        )
    end
    return BVProblem(
        f, interior ? device_interior_bc! : (iip ? device_bc! : device_bc),
        u0, (zero(T), one(T)), p
    )
end

function packed_init(prob, alg; kwargs...)
    return FIRK.__init_firk_device(prob, alg, FIRK.__device_initial_state(prob.u0, prob.p, first(prob.tspan)); kwargs...)
end

function test_device_backend(upload, is_device, platform; gpu = false)
    @testset "Packed FIRK $Alg iip=$iip two-point=$twopoint" for Alg in
            (
                RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7,
                LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5,
                LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5,
                LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5,
            ),
            (iip, twopoint) in ((true, false), (false, true))
        prob = device_problem(upload, Float64, iip, twopoint)
        alg = Alg(; platform, jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff())))
        cache = gpu ? init(prob, alg; dt = 0.05, adaptive = false) :
            packed_init(prob, alg; dt = 0.05, adaptive = false)
        @test is_device(BoundaryValueDiffEqFIRK.__firk_states(cache))
        @test is_device(cache.residual)
        @test BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache) !== nothing
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        tol = Alg == RadauIIa1 ? 0.06 : 0.003
        @test maximum(abs, Array(sol(0.37)) .- exp(0.37)) < tol
        @test is_device(sol.u[1])
        @test is_device(sol(0.37))
        @test sol(0.0; idxs = 1) ≈ 1 atol = 1.0e-6
        @test sol(1.0; idxs = 1) ≈ exp(1.0) atol = 1.0e-6
    end

    @testset "AD and interior boundary conditions $mode" for mode in
        (AutoForwardDiff(; chunksize = 3), AutoFiniteDiff(), AutoFiniteDiff(; fdjtype = Val(:central)))
        prob = device_problem(upload, Float64, true, false; interior = true)
        cache = packed_init(
            prob, RadauIIa3(; platform, jac_alg = BVPJacobianAlgorithm(AutoSparse(mode)));
            dt = 0.1, adaptive = false
        )
        @test !BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache).boundary_fallback
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test maximum(abs, Array(sol(0.51)) .- exp(0.51)) < 1.0e-4
        @test maximum(abs, Array(sol(0.51, Val{1})) .- exp(0.51)) < 1.0e-3
        out = upload(zeros(2))
        sol(out, 0.51)
        @test Array(out) ≈ Array(sol(0.51))
        @test length(sol([0.2, 0.7]).u) == 2
    end

    return @testset "Adaptive defect control and reuse" begin
        prob = device_problem(upload, Float64, true, false)
        cache = packed_init(
            prob, RadauIIa3(;
                platform,
                jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff()))
            );
            dt = 0.5, abstol = 1.0e-7, adaptive = true
        )
        initial_nodes = length(cache.host_mesh)
        sol = solve!(cache)
        @test successful_retcode(sol.retcode)
        @test length(sol.t) > initial_nodes
        @test maximum(FIRK.__firk_device_defect!(cache)) <= 1.0e-7
        @test size(BoundaryValueDiffEqFIRK.__firk_jacobian(cache), 2) == length(BoundaryValueDiffEqFIRK.__firk_states(cache))
        @test BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache).pattern isa SparseMatrixCSC
        @test maximum(abs, Array(sol(0.37)) .- exp(0.37)) < 1.0e-5
        old = Array(sol(0.37))
        fill!(BoundaryValueDiffEqFIRK.__firk_states(cache), 0.7)
        again = solve!(cache)
        @test successful_retcode(again.retcode)
        @test Array(sol(0.37)) == old
        @test Array(again(0.37)) ≈ old atol = 1.0e-6
        limited = packed_init(
            prob, RadauIIa3(; platform, max_num_subintervals = 2);
            dt = 0.5, abstol = 1.0e-12, adaptive = true
        )
        @test !successful_retcode(solve!(limited))

        # A failed nonlinear iteration takes the same recovery path as a
        # rejected defect estimate: bisect, interpolate on-device, rebuild the
        # mesh-sized buffers, then retry until the interval limit is reached.
        failed = packed_init(
            prob, RadauIIa3(; platform, max_num_subintervals = 8);
            dt = 0.25, adaptive = true, nlsolve_kwargs = (; maxiters = 0)
        )
        original_mesh = copy(failed.host_mesh)
        failed_sol = solve!(failed)
        @test !successful_retcode(failed_sol.original.retcode)
        @test failed_sol.retcode == ReturnCode.MaxIters
        @test length(failed_sol.t) == 2 * (length(original_mesh) - 1) + 1
        @test failed.host_mesh == failed_sol.t
        @test size(BoundaryValueDiffEqFIRK.__firk_states(failed), 2) == (length(failed_sol.t) - 1) * (failed.TU.s + 1) + 1
    end
end

if !isdefined(@__MODULE__, :FIRK_GPU_TESTS)
    test_device_backend(identity, x -> x isa Array, CPU())
    @testset "Packed sparse Jacobian agrees with independent ForwardDiff" begin
        prob = device_problem(identity, Float64, true, false; interior = true)
        cache = packed_init(prob, RadauIIa3(); dt = 0.2, adaptive = false)
        u = vec(copy(BoundaryValueDiffEqFIRK.__firk_states(cache)))
        u .+= range(0.01, 0.1; length = length(u))
        J = copy(BoundaryValueDiffEqFIRK.__firk_jacobian(cache))
        FIRK.__device_jacobian!(J, u, cache)
        reference = ForwardDiff.jacobian(u) do x
            residual = similar(x, length(cache.residual))
            FIRK.__device_residual!(residual, x, cache)
        end
        @test Matrix(J) ≈ reference atol = 1.0e-10
        @test nnz(J) < length(J)
        @test maximum(g.ncolors for g in BoundaryValueDiffEqFIRK.__firk_jacobian_plan(cache).groups) < length(u)
    end
end
