import DiffEqDiffTools, NLsolve

mutable struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end
(p::BVPJacobianWrapper)(resid,u) = p.loss(resid,u)
(p::BVPJacobianWrapper)(u) = (resid = similar(u); p.loss(resid,u); resid)

function ConstructJacobian(f!::BVPJacobianWrapper, S::BVPSystem, y)
    jac_cache = DiffEqDiffTools.JacobianCache(
                                  similar(y),similar(y),similar(y))
    function fj!(F::Vector, J, x::Vector)
        DiffEqDiffTools.finite_difference_jacobian!(J, f!, x, jac_cache)
        f!(F,x)
    end
    j!(J, x::Array)  = (F = jac_cache.fx; fj!(F, J, x))
    J0 = BandedMatrix(Zeros{eltype(S.y[1])}(S.M*S.N, S.M*S.N), (S.M-1, S.M-1))
    NLsolve.OnceDifferentiable(f!.loss, j!, fj!, jac_cache.x1, jac_cache.fx, sparse(J0))
end

function ConstructJacobian(f!::BVPJacobianWrapper, y)
    jac_cache = DiffEqDiffTools.JacobianCache(
                                    similar(y),similar(y),similar(y))
    function fj!(F, J, x::Vector)
        DiffEqDiffTools.finite_difference_jacobian!(J, f!, x, jac_cache)
        f!(F,x)
    end
    j!(J::Array, x::Vector)  = (F = jac_cache.fx; fj!(F, J, x))
    NLsolve.OnceDifferentiable(f!.loss, j!, fj!, jac_cache.x1, jac_cache.fx)
end
