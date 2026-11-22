nested_nonlinear!(du, u, p, t) = (du[1] = u[1]^2; nothing)
nested_interior!(r, u, p, t) = (r[1] = u(0.23)[1] - 1 / 1.77; nothing)

function test_nested_device_jacobian(upload, platform)
    return @testset "Implicit stage Jacobian and interior boundary conditions" begin
        prob = BVProblem(nested_nonlinear!, nested_interior!, upload([0.6]), (0.0, 1.0))
        cache = packed_init(
            prob, RadauIIa3(;
                platform, nested_nlsolve = true, nested_nlsolve_kwargs = (; abstol = 1.0e-13),
                jac_alg = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff(chunksize = 1)))
            ); dt = 0.2, adaptive = false
        )
        u, J = vec(copy(FIRK.__firk_unknowns(cache))), copy(FIRK.__firk_jacobian(cache))
        FIRK.__device_jacobian!(J, u, cache)
        reference = zeros(length(cache.residual), length(u))
        x = Array(u)
        for column in eachindex(x)
            plus, minus = copy(x), copy(x)
            plus[column] += 1.0e-5
            minus[column] -= 1.0e-5
            rp, rm = similar(cache.residual), similar(cache.residual)
            FIRK.__device_residual!(rp, upload(plus), cache)
            FIRK.__device_residual!(rm, upload(minus), cache)
            reference[:, column] .= (Array(rp) .- Array(rm)) ./ 2.0e-5
        end
        @test Matrix(SparseMatrixCSC(J)) ≈ reference atol = 1.0e-7
        # Differentiation must still work after warm starts converge in zero steps.
        FIRK.__device_jacobian!(J, u, cache)
        @test Matrix(SparseMatrixCSC(J)) ≈ reference atol = 1.0e-7
    end
end
