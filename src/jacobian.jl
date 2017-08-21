import DiffEqDiffTools, NLsolve

mutable struct BVPJacobianWrapper{LossType,CacheType} <: Function
    loss::LossType
    x1::CacheType
    fx1::CacheType
end

(p::BVPJacobianWrapper)(resid,u) = p.loss(u,resid)

(p::BVPJacobianWrapper)(u) = (resid = similar(u); p.loss(u,resid); resid)

function ConstructDifferentiableMultivariateFunction(f!::BVPJacobianWrapper)
    fg!(x::Vector, fx::Vector, gx) = DiffEqDiffTools.finite_difference_jacobian!(gx, f!, x, Val{:central}, fx, Val{:JacobianWrapper})
    g!(x::Vector, gx::Array)  = (fx = similar(x); fg!(x, fx, gx))
    # J = bzeros(eltype(S.y[1]), S.M*S.N, S.M*S.N, S.M, S.M)
    NLsolve.DifferentiableMultivariateFunction(f!.loss, g!, fg!)
end
