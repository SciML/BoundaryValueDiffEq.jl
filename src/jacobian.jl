struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end
(jw::BVPJacobianWrapper)(resid, u, p) = jw.loss(resid, u, p)
(jw::BVPJacobianWrapper)(u, p) = (resid = similar(u); jw.loss(resid, u, p); resid)

function _construct_nonlinear_problem_with_jacobian(f!::BVPJacobianWrapper, S::BVPSystem,
    y, p)
    J0 = BandedMatrix(similar(first(S.y), (S.M * S.N, S.M * S.N)), (S.M, S.M))
    jac_cache = FiniteDiff.JacobianCache(similar(y),
        similar(y),
        similar(y);
        colorvec = ArrayInterface.matrix_colors(J0),
        sparsity = sparse(J0))
    function jac!(J, x, p)
        F = jac_cache.fx
        _f!(F, x) = f!(F, x, p)
        FiniteDiff.finite_difference_jacobian!(J, _f!, x, jac_cache)
        return nothing
    end
    return NonlinearProblem(NonlinearFunction{true}(f!; jac = jac!), y, p)
end
