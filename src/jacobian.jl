using FiniteDiff: FiniteDiff
using NLsolve: NLsolve

mutable struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end
(p::BVPJacobianWrapper)(resid, u) = p.loss(resid, u)
(p::BVPJacobianWrapper)(u) = (resid = similar(u); p.loss(resid, u); resid)

function ConstructJacobian(f!::BVPJacobianWrapper, S::BVPSystem, y)
    jac_cache = FiniteDiff.JacobianCache(similar(y), similar(y), similar(y))
    function fj!(F, J, x)
        FiniteDiff.finite_difference_jacobian!(J, f!, x, jac_cache)
        return f!(F, x)
    end
    j!(J, x) = (F = jac_cache.fx; fj!(F, J, x))
    J0 = BandedMatrix(Zeros{eltype(S.y[1])}(S.M * S.N, S.M * S.N), (S.M - 1, S.M - 1))
    return NLsolve.OnceDifferentiable(f!.loss, j!, fj!, jac_cache.x1, jac_cache.fx, sparse(J0))
end

function ConstructJacobian(f!::BVPJacobianWrapper, y)
    jac_cache = FiniteDiff.JacobianCache(similar(y), similar(y), similar(y))
    function fj!(F, J, x)
        FiniteDiff.finite_difference_jacobian!(J, f!, x, jac_cache)
        return f!(F, x)
    end
    j!(J, x) = (F = jac_cache.fx; fj!(F, J, x))
    return NLsolve.OnceDifferentiable(f!.loss, j!, fj!, jac_cache.x1, jac_cache.fx)
end
