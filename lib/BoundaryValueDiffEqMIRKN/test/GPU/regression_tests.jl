f!(a, v, u, p, t) = (a[1, 1] = u[1, 1]; a[1, 2] = u[1, 2]; nothing)
function bc!(r, v, u, p, t)
    r[1] = u(t[1])[1, 1] - 1
    r[2] = u[1, 2, 1] - 1
    r[3] = v[1, 1, end] - exp(t[end])
    r[4] = v[:, :, end][1, 2] - exp(t[end])
    return nothing
end
@testset "CUDA matrix indexing and interpolation" begin
    p = SecondOrderBVProblem(f!, bc!, CuArray(ones(1, 2)), (0.0, 1.0))
    for Alg in (MIRKN4, MIRKN6)
        c = init(p, Alg(); dt = 0.1)
        @test !BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(c).boundary_fallback
        s = solve!(c)
        @test successful_retcode(s)
        @test Array(s.u[end].x[1]) ≈ fill(exp(1), 1, 2) atol = 1.0e-5
        out = similar(s.u[1]); s(out, 0.37)
        @test Array(out.x[1]) ≈ Array(s(0.37).x[1])
        @test Array(s(0.37, Val{1}).x[1]) ≈ fill(exp(0.35), 1, 2) atol = 0.003
        @test length(s([0.2, 0.3]).u) == 2
    end
end
@testset "CUDA mixed sparse AD" begin
    p = SecondOrderBVProblem(f!, bc!, CuArray(ones(1, 2)), (0.0, 1.0))
    a = MIRKN6(jac_alg = BVPJacobianAlgorithm(bc_diffmode = AutoFiniteDiff(fdjtype = Val(:central)), nonbc_diffmode = AutoSparse(AutoForwardDiff(chunksize = 3))))
    c = init(p, a; dt = 0.1)
    @test length(BoundaryValueDiffEqMIRKN.__mirkn_jacobian_plan(c).groups) == 2
    @test successful_retcode(solve!(c))
end
@testset "CUDA underdetermined NLLS" begin
    ff!(a, v, u, p, t) = (a[1] = u[1]; nothing)
    b!(r, v, u, p, t) = (r[1] = u(t[1])[1] - 1; nothing)
    for nlls in (nothing, Val(true))
        p = SecondOrderBVProblem(DynamicalBVPFunction(ff!, b!; bcresid_prototype = zeros(1)), CuArray([0.8]), (0.0, 1.0); nlls)
        c = init(p, MIRKN6(); dt = 0.1)
        s = solve!(c)
        @test successful_retcode(s)
        @test maximum(abs, s.resid) < 1.0e-5
    end
end
