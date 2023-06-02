mutable struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end
(jw::BVPJacobianWrapper)(resid, u, p) = jw.loss(resid, u, p)
(jw::BVPJacobianWrapper)(u, p) = (resid = similar(u); jw.loss(resid, u, p); resid)

function _construct_nonlinear_problem_with_jacobian(f!::BVPJacobianWrapper, S::BVPSystem,
                                                    y, p)
    jac_cache = FiniteDiff.JacobianCache(similar(y), similar(y), similar(y))
    function jac!(J, x, p)
        F = jac_cache.fx
        _f!(F, x) = f!(F, x, p)
        FiniteDiff.finite_difference_jacobian!(J, _f!, x, jac_cache)
        return nothing
    end
    J0 = BandedMatrix(Zeros{eltype(first(S.y))}(S.M * S.N, S.M * S.N), (S.M - 1, S.M - 1))
    nlf = NonlinearFunction{true}(f!.loss; jac = jac!, jac_prototype = sparse(J0))
    return NonlinearProblem(nlf, y, p)
end
