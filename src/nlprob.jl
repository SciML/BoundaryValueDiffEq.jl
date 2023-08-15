function construct_MIRK_nlproblem(S::BVPSystem, prob::BVProblem, TU, cache, mesh, y)
    function loss_bc!(resid, u)
        u_ = reshape(u, S.M, S.N)
        eval_bc_residual!(resid, prob.problem_type, S, u_, prob.p, mesh)
    end

    @views function loss_collocation!(resid, u)
        u_ = reshape(u, S.M, S.N)
        resid_ = reshape(resid, S.M, S.N - 1)
        Φ!(resid_, S, TU, cache, u_, prob.p, mesh)
    end

    @views function loss!(resid, u, p)
        u_ = reshape(u, S.M, S.N)
        resid_ = reshape(resid, S.M, S.N)
        Φ!(resid_[:, 2:end], S, TU, cache, u_, p, mesh)
        eval_bc_residual!(resid_[:, 1], prob.problem_type, S, u_, p, mesh)
        return resid
    end

    resid = similar(y)
    resid₁, resid₂ = @view(resid[1:(S.M)]), @view(resid[(S.M + 1):end])

    ad₁ = AutoFiniteDiff()
    sd₁ = NoSparsityDetection()
    cache₁ = sparse_jacobian_cache(ad₁, sd₁, loss_bc!, resid₁, y)

    ad₂ = AutoSparseFiniteDiff()
    Jₛ = sparse(BandedMatrix(similar(y, (S.M * (S.N - 1), S.M * S.N)), (1, 2S.M)))
    sd₂ = JacPrototypeSparsityDetection(; jac_prototype = Jₛ)
    cache₂ = sparse_jacobian_cache(ad₂, sd₂, loss_collocation!, resid₂, y)

    function jac!(J, x, p)
        sparse_jacobian!(@view(J[1:(S.M), :]), ad₁, cache₁, loss_bc!, resid₁, x)
        sparse_jacobian!(@view(J[(S.M + 1):end, :]), ad₂, cache₂, loss_collocation!, resid₂,
            x)
        return J
    end

    return NonlinearProblem(NonlinearFunction{true}(loss!; jac = jac!), y, prob.p)
end
