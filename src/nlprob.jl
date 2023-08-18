function construct_MIRK_nlproblem(S::BVPSystem, prob::BVProblem, TU, cache, mesh, y,
    jac_alg::MIRKJacobianComputationAlgorithm)
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
    resid_bc, resid_collocation = @view(resid[1:(S.M)]), @view(resid[(S.M + 1):end])

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
            NoSparsityDetection()
    cache_bc = sparse_jacobian_cache(jac_alg.bc_diffmode, sd_bc, loss_bc!, resid_bc, y)

    sd_collocation = if jac_alg.collocation_diffmode isa AbstractSparseADType
        Jₛ = sparse(BandedMatrix(similar(y, (S.M * (S.N - 1), S.M * S.N)), (1, 2 * S.M)))
        JacPrototypeSparsityDetection(; jac_prototype = Jₛ)
    else
        NoSparsityDetection()
    end
    cache_collocation = sparse_jacobian_cache(jac_alg.collocation_diffmode, sd_collocation,
        loss_collocation!, resid_collocation, y)

    jac_prototype = vcat(SparseDiffTools.__init_𝒥(cache_bc),
        SparseDiffTools.__init_𝒥(cache_collocation))

    function jac!(J, x, p)
        sparse_jacobian!(@view(J[1:(S.M), :]), jac_alg.bc_diffmode, cache_bc, loss_bc!,
            resid_bc, x)
        sparse_jacobian!(@view(J[(S.M + 1):end, :]), jac_alg.collocation_diffmode,
            cache_collocation, loss_collocation!, resid_collocation, x)
        return J
    end

    return NonlinearProblem(NonlinearFunction{true}(loss!; jac, jac_prototype), y, prob.p)
end
