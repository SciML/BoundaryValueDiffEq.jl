using BoundaryValueDiffEqMIRKN, Test, SparseArrays, LinearAlgebra
using LinearSolve: KrylovJL_LSMR
using StaticArrays: SVector
const MN = BoundaryValueDiffEqMIRKN

function exp_rhs!(a, v, u, p, t)
    for j in eachindex(a)
        a[j] = p[1] * v[j] + (1 - p[1]) * u[j]
    end
    return nothing
end
exp_rhs(v, u, p, t) = SVector(p[1] * v[1] + (1 - p[1]) * u[1], p[1] * v[2] + (1 - p[1]) * u[2])
function exp_bc!(r, v, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = u[:, 1][2] - 1
    r[3] = v(t[end])[1] - exp(t[end])
    r[4] = v.u[end][2] - exp(t[end])
    return nothing
end
exp_bc(v, u, p, t) = SVector(
    u(t[1])[1] - 1, u[:, 1][2] - 1,
    v(t[end])[1] - exp(t[end]), v.u[end][2] - exp(t[end])
)
exp_left!(r, v, u, p) = (r[1] = u[1] - 1; nothing)
exp_right!(r, v, u, p) = (r[1] = v[1] - exp(one(eltype(u))); r[2] = u[2] - exp(one(eltype(u))); r[3] = v[2] - exp(one(eltype(u))); nothing)
exp_left(v, u, p) = SVector(u[1] - 1)
exp_right(v, u, p) = SVector(v[1] - exp(one(eltype(u))), u[2] - exp(one(eltype(u))), v[2] - exp(one(eltype(u))))

function test_device_backend(upload, platform; gpu = false)
    return @testset "Resident $Alg $T iip=$iip two=$two matrix=$matrix" for Alg in (MIRKN4, MIRKN6),
            T in (Float32, Float64), iip in (true, false), two in (true, false), matrix in (true, false)
        u0 = upload(matrix ? reshape(T[0.8, 1.2], 1, 2) : T[0.8, 1.2])
        f = iip ? exp_rhs! : exp_rhs
        bc = two ? (iip ? (exp_left!, exp_right!) : (exp_left, exp_right)) : (iip ? exp_bc! : exp_bc)
        prob = two ? TwoPointSecondOrderBVProblem(
                f, bc, u0, (zero(T), one(T)), upload(T[0.4]);
                bcresid_prototype = (upload(zeros(T, 1)), upload(zeros(T, 3))), nlls = Val(false)
            ) :
            SecondOrderBVProblem(f, bc, u0, (zero(T), one(T)), upload(T[0.4]))
        alg = Alg()
        cache = gpu ? init(prob, alg; dt = T(0.1), abstol = T(1.0e-6)) :
            MN.__init_mirkn_device(
                prob, alg, u0; dt = T(0.1), abstol = T(1.0e-6), adaptive = false,
                controller = NoErrorControl(), nlsolve_kwargs = (; abstol = T(1.0e-6)), optimize_kwargs = (;), verbose = false
            )
        @test cache.alg.platform isa typeof(platform)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test sol.u[1].x[1] isa typeof(u0)
        @test sol.u[1].x[2] isa typeof(u0)
        @test sol.original.u isa typeof(vec(u0))
        @test maximum(norm(Array(s.x[1]) .- exp(t), Inf) for (s, t) in zip(sol.u, sol.t)) < 1.0e-4
        @test maximum(norm(Array(s.x[2]) .- exp(t), Inf) for (s, t) in zip(sol.u, sol.t)) < 1.0e-4
        @test size(sol(0.37).x[1]) == size(u0)
        # MIRKN exposes linear interpolation between its high order mesh values.
        @test Array(sol(0.37).x[1]) ≈ fill(exp(0.37), size(u0)) atol = 0.003
        @test sol(0.37; idxs = 1) ≈ exp(0.37) atol = 0.003
        @test Array(sol(0.37; idxs = [1, 3])) ≈ fill(exp(0.37), 2) atol = 0.003
        if gpu
            @test SparseArrays.nonzeros(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache)) isa typeof(vec(u0))
            @test !BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).boundary_fallback
            @test occursin("CuSparseMatrixCSR", string(typeof(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache))))
        end
    end
end

function test_device_features(upload, platform; gpu = false)
    function make_cache(prob, alg; kwargs...)
        gpu && return init(prob, alg; dt = 0.1, kwargs...)
        u0 = MN.__device_initial_state(prob.u0, prob.p, 0.0)
        # Exercise the resident pipeline with the same Newton/LSMR algorithms
        # used on CUDA, without compiling the CPU trust-region polyalgorithm
        # for every combination of initial guess and differentiation mode.
        nbc = prod(first(MN.__device_bc_sizes(prob, u0)))
        nlsolve = nbc == 2length(u0) ? NewtonRaphson() : GaussNewton(; linsolve = KrylovJL_LSMR())
        MN.@set! alg.nlsolve = nlsolve
        return MN.__init_mirkn_device(
            prob, alg, u0;
            dt = 0.1, abstol = 1.0e-8, adaptive = false, controller = NoErrorControl(),
            nlsolve_kwargs = (; abstol = 1.0e-8), optimize_kwargs = (;), verbose = false, kwargs...
        )
    end
    f!(a, v, u, p, t) = (a[1] = u[1]; nothing)
    bc!(r, v, u, p, t) = (r[1] = u(t[1])[1] - 1; r[2] = v(t[end])[1] - exp(t[end]); nothing)
    @testset "Initial guesses and differentiation modes" begin
        base = upload([0.8])
        guesses = (base, [copy(base) for _ in 1:11], (p, t) -> copy(base))
        modes = (
            AutoForwardDiff(; chunksize = 3), AutoSparse(AutoForwardDiff(; chunksize = 2)),
            AutoFiniteDiff(), AutoFiniteDiff(; fdjtype = Val(:central)),
        )
        for Alg in (MIRKN4, MIRKN6), guess in guesses, mode in modes
            prob = SecondOrderBVProblem(f!, bc!, guess, (0.0, 1.0))
            cache = make_cache(prob, Alg(; jac_alg = BVPJacobianAlgorithm(mode)))
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test Array(sol.u[end].x[1]) ≈ [exp(1)] atol = 5.0e-5
        end
    end
    @testset "Rectangular sparse least squares" begin
        function extra_bc!(r, v, u, p, t)
            r[1] = u(t[1])[1] - 1
            r[2] = v(t[1])[1] - 1
            r[3] = u(t[end])[1] - exp(t[end])
            return nothing
        end
        for Alg in (MIRKN4, MIRKN6)
            fun = DynamicalBVPFunction(f!, extra_bc!; bcresid_prototype = zeros(3))
            prob = SecondOrderBVProblem(fun, upload([0.8]), (0.0, 1.0); nlls = Val(true))
            cache = make_cache(prob, Alg())
            @test size(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache), 1) == size(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache), 2) + 1
            nl = MN.__construct_nlproblem(cache, vec(BoundaryValueDiffEqMIRKN.__mirkn_states(cache)))
            @test nl isa MN.SciMLBase.NonlinearLeastSquaresProblem
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test Array(sol.u[end].x[1]) ≈ [exp(1)] atol = 1.0e-4
        end
    end
    return @testset "Cache reuse rebuilds boundary dependencies" begin
        function moving_bc!(r, v, u, p, t)
            r[1] = u(t[1])[1] - 1
            r[2] = v(p[1])[1] - exp(p[1])
            return nothing
        end
        p = upload([0.5])
        prob = SecondOrderBVProblem(f!, moving_bc!, upload([0.8]), (0.0, 1.0), p)
        cache = make_cache(prob, MIRKN6())
        first_sol = solve!(cache)
        old = copy(BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).pattern)
        snapshot = Array(first_sol.u[end].x[1])
        copyto!(p, [1.0])
        second_sol = solve!(cache)
        @test successful_retcode(first_sol) && successful_retcode(second_sol)
        @test BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).pattern != old
        @test Array(first_sol.u[end].x[1]) == snapshot
        @test Array(second_sol.u[end].x[1]) ≈ [exp(1)] atol = 5.0e-5
    end
end

function test_hybrid(platform)
    return @testset "Hybrid collocation $Alg iip=$iip two=$two" for Alg in (MIRKN4, MIRKN6), iip in (true, false), two in (true, false)
        f = iip ? exp_rhs! : exp_rhs
        bc = two ? (iip ? (exp_left!, exp_right!) : (exp_left, exp_right)) : (iip ? exp_bc! : exp_bc)
        prob = two ? TwoPointSecondOrderBVProblem(
                f, bc, [0.8, 1.2], (0.0, 1.0), [0.4];
                bcresid_prototype = (zeros(1), zeros(3))
            ) :
            SecondOrderBVProblem(f, bc, [0.8, 1.2], (0.0, 1.0), [0.4])
        # Legacy CPU MIRKN expects one M-sized block for each boundary.
        if two
            bc = iip ? (
                    (r, v, u, p) -> (r[1] = u[1] - 1; r[2] = u[2] - 1; nothing),
                    (r, v, u, p) -> (r[1] = v[1] - exp(1); r[2] = v[2] - exp(1); nothing),
                ) :
                ((v, u, p) -> SVector(u[1] - 1, u[2] - 1), (v, u, p) -> SVector(v[1] - exp(1), v[2] - exp(1)))
            prob = TwoPointSecondOrderBVProblem(f, bc, [0.8, 1.2], (0.0, 1.0), [0.4]; bcresid_prototype = (zeros(2), zeros(2)))
        end
        for mode in (AutoSparse(AutoForwardDiff()), AutoSparse(AutoFiniteDiff()))
            alg = Alg(; platform, jac_alg = BVPJacobianAlgorithm(mode))
            cache = init(prob, alg; dt = 0.1)
            if platform isa CPU
                cache.device_cache === nothing || error("Unexpected CPU offload")
                # A separate constructor exercises packing/transfers on normal CI.
                # Production hybrid dispatch is tested on CUDA below.
                continue
            end
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test sol.u[end].x[1] ≈ fill(exp(1), 2) atol = 1.0e-4
        end
    end
end
