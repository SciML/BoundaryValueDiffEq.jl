import DiffEqDiffTools, NLsolve

mutable struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end
(p::BVPJacobianWrapper)(resid,u) = p.loss(resid,u)
(p::BVPJacobianWrapper)(u) = (resid = similar(u); p.loss(resid,u); resid)

function ConstructSparseJacobian(f!::BVPJacobianWrapper, S::BVPSystem, y)
    jac_cache = DiffEqDiffTools.JacobianCache(
                                  similar(y),similar(y),similar(y))
    function fg!(x::Vector, fx::Vector, gx)
        DiffEqDiffTools.finite_difference_jacobian!(gx, f!, x, jac_cache)
        f!(fx,x)
    end
    g!(gx::Array, x::Vector)  = (fx = similar(x); fg!(fx, gx, x))
    J = bzeros(eltype(S.y[1]), S.M*S.N, S.M*S.N, S.M-1, S.M-1)
    NLsolve.OnceDifferentiable(f!.loss, g!, fg!, sparse(J))
end

function ConstructJacobian(f!::BVPJacobianWrapper, S::BVPSystem, y)
    jac_cache = DiffEqDiffTools.JacobianCache(
                                    similar(y),similar(y),similar(y))
    function fg!(fx::Vector, gx, x::Vector)
        DiffEqDiffTools.finite_difference_jacobian!(gx, f!, x, jac_cache)
        f!(fx,x)
    end
    g!(gx::Array, x::Vector)  = (fx = similar(x); fg!(fx, gx, x))
    NLsolve.OnceDifferentiable(f!.loss, g!, fg!, y)
end
