using BoundaryValueDiffEqMIRKN, Test, SparseArrays, LinearAlgebra
using StaticArrays: SVector
const MN = BoundaryValueDiffEqMIRKN
const FD = MN.ForwardDiff

function resident_init(prob, alg; dt = 0.125, kwargs...)
    u0 = MN.__device_initial_state(prob.u0, prob.p, first(prob.tspan))
    return MN.__init_mirkn_device(
        prob, alg, u0; dt, adaptive = false, abstol = 1.0e-9,
        controller = NoErrorControl(), nlsolve_kwargs = (; abstol = 1.0e-9),
        optimize_kwargs = (;), verbose = false, kwargs...
    )
end

function coupled!(ddu, du, u, p, t)
    ddu[1] = sin(u[1]) + du[2] * u[2] + t
    ddu[2] = du[1]^2 - 2u[2] + p[1]
    return nothing
end
coupled(du, u, p, t) = SVector(sin(u[1]) + du[2] * u[2] + t, du[1]^2 - 2u[2] + p[1])
function multipoint!(r, du, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = du(t[end])[2] - 2
    r[3] = u(0.37)[2] + du(0.73)[1]
    r[4] = u[:, end][1] * du.u[1][2]
    return nothing
end
multipoint(du, u, p, t) = (
    u(t[1])[1] - 1, du(t[end])[2] - 2,
    u(0.37)[2] + du(0.73)[1], u[:, end][1] * du.u[1][2],
)
left!(r, du, u, p) = (r[1] = u[1] + du[2]; nothing)
right!(r, du, u, p) = (r[1] = u[1]; r[2] = du[1] * u[2]; r[3] = du[2]; nothing)
left(du, u, p) = (u[1] + du[2],)
right(du, u, p) = (u[1], du[1] * u[2], du[2])

@testset "Packed MIRKN residuals and colored Jacobians" begin
    @testset "$Alg $T iip=$iip two=$two matrix=$matrix" for Alg in (MIRKN4, MIRKN6),
            T in (Float32, Float64), iip in (true, false), two in (true, false), matrix in (true, false)
        u0 = matrix ? reshape(T[0.3, 0.7], 1, 2) : T[0.3, 0.7]
        f = iip ? coupled! : coupled
        bc = two ? (iip ? (left!, right!) : (left, right)) : (iip ? multipoint! : multipoint)
        prob = two ? TwoPointSecondOrderBVProblem(
                f, bc, u0, (zero(T), one(T)), T[0.4];
                bcresid_prototype = (zeros(T, 1), zeros(T, 3))
            ) :
            SecondOrderBVProblem(f, bc, u0, (zero(T), one(T)), T[0.4])
        for mode in (AutoForwardDiff(; chunksize = 3), AutoFiniteDiff(), AutoFiniteDiff(; fdjtype = Val(:central)))
            alg = Alg(; jac_alg = BVPJacobianAlgorithm(mode))
            cache = resident_init(prob, alg; dt = T(0.25))
            y = BoundaryValueDiffEqMIRKN.__mirkn_states(cache)
            y .+= reshape(T.(range(0, 0.5; length = length(y))), size(y))
            u = vec(y)
            residual = similar(cache.residual)
            MN.__device_residual!(residual, u, cache)
            # Compare with the original CPU collocation equations, including dy.
            cpu = init(prob, alg; dt = T(0.25))
            nodes, M = size(y, 2), cache.M
            flat = vcat(vec(y[1:M, :]), vec(y[(M + 1):2M, :]))
            states = MN.recursive_unflatten!(cpu.y, flat)
            expected = if iip
                r = [zeros(T, M) for _ in 1:(2(nodes - 1))]
                MN.Φ!(r, cpu, states, flat)
                r
            else
                MN.Φ(cpu, states, flat)
            end
            packed = reduce(vcat, [vcat(expected[i], expected[nodes - 1 + i]) for i in 1:(nodes - 1)])
            nleft = prod(cache.resid_size[1])
            tol = T === Float32 ? 3.0f-5 : 1.0e-12
            @test residual[(nleft + 1):(nleft + length(packed))] ≈ packed rtol = tol atol = tol
            J = BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache)
            MN.__device_jacobian!(J, u, cache)
            ref = FD.jacobian(u) do x
                out = similar(x, length(residual))
                MN.__device_residual!(out, x, cache)
                out
            end
            jtol = mode isa AutoFiniteDiff ? (T === Float32 ? 0.008 : 2.0e-6) : tol
            @test Matrix(J) ≈ ref rtol = jtol atol = jtol
            @test !BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).boundary_fallback
            products = MN.__device_jacobian_products(cache)
            v = fill(T(0.3), length(u))
            jv = similar(residual)
            products.jvp(jv, v, u, cache.p)
            @test jv ≈ ref * v rtol = jtol atol = jtol
            jt = similar(u)
            products.vjp(jt, fill(T(0.2), length(residual)), u, cache.p)
            @test jt ≈ ref' * fill(T(0.2), length(residual)) rtol = jtol atol = jtol
        end
    end
end

@testset "Sparse structure scales with intervals" begin
    f!(a, v, u, p, t) = (a[1] = u[1]; nothing)
    bc!(r, v, u, p, t) = (r[1] = u(t[1])[1]; r[2] = v(t[end])[1]; nothing)
    prob = SecondOrderBVProblem(f!, bc!, [1.0], (0.0, 1.0))
    for Alg in (MIRKN4, MIRKN6), n in (8, 64, 512)
        cache = resident_init(prob, Alg(); dt = 1 / n)
        @test nnz(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache)) == 8n + 2
        @test maximum(g.ncolors for g in BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).groups) <= 4
    end
end

@testset "Mixed modes, known sparsity and conservative fallback" begin
    f!(a, v, u, p, t) = (a[1] = u[1] + 0.1v[1]; nothing)
    function branch!(r, v, u, p, t)
        r[1] = u(t[1])[1] > 0 ? u(t[end])[1] : v(t[end])[1]
        r[2] = u(0.37)[1]
        return nothing
    end
    prob = SecondOrderBVProblem(f!, branch!, [1.0], (0.0, 1.0))
    alg = MIRKN4(;
        jac_alg = BVPJacobianAlgorithm(;
            bc_diffmode = AutoFiniteDiff(; fdjtype = Val(:central)),
            nonbc_diffmode = AutoForwardDiff(; chunksize = 3)
        )
    )
    cache = resident_init(prob, alg)
    @test BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).boundary_fallback
    @test length(BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(cache).groups) == 2
    MN.__device_jacobian!(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache), vec(BoundaryValueDiffEqMIRKN.__mirkn_states(cache)), cache)
    ref = FD.jacobian(vec(BoundaryValueDiffEqMIRKN.__mirkn_states(cache))) do x
        r = similar(x)
        MN.__device_residual!(r, x, cache)
    end
    @test Matrix(BoundaryValueDiffEqMIRKN.__mirkn_jacobian(cache)) ≈ ref atol = 1.0e-7
    pattern = sparse(ones(2, length(BoundaryValueDiffEqMIRKN.__mirkn_states(cache))))
    mode = AutoSparse(AutoForwardDiff(); sparsity_detector = MN.ADTypes.KnownJacobianSparsityDetector(pattern))
    known = resident_init(prob, MIRKN6(; jac_alg = BVPJacobianAlgorithm(; bc_diffmode = mode)))
    @test !BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(known).boundary_fallback
    @test BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(known).pattern[1:2, :] == pattern
end
