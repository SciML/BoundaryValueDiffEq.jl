import DiffEqDiffTools, NLsolve

mutable struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end

(p::BVPJacobianWrapper)(resid,u) = p.loss(u,resid)

(p::BVPJacobianWrapper)(u) = (resid = similar(u); p.loss(u,resid); resid)

function ConstructSparseJacobian(f!::BVPJacobianWrapper, S::BVPSystem, y)
    RealOrComplex = (eltype(y) <: Complex) ? Val{:Complex} : Val{:Real}
    jac_cache = DiffEqDiffTools.JacobianCache(Val{:central},RealOrComplex,
                                  similar(y),similar(y),similar(y))
    function fg!(x::Vector, fx::Vector, gx)
        DiffEqDiffTools.finite_difference_jacobian!(gx, f!, x, jac_cache)
    end
    g!(x::Vector, gx::Array)  = (fx = similar(x); fg!(x, fx, gx))
    J = bzeros(eltype(S.y[1]), S.M*S.N, S.M*S.N, S.M-1, S.M-1)
    NLsolve.DifferentiableMultivariateFunction(f!.loss, g!, fg!, sparse(J))
end

function ConstructJacobian(f!::BVPJacobianWrapper, S::BVPSystem, y)
    RealOrComplex = (eltype(y) <: Complex) ? Val{:Complex} : Val{:Real}
    jac_cache = DiffEqDiffTools.JacobianCache(Val{:central},RealOrComplex,
                                  similar(y),similar(y),similar(y))
    function fg!(x::Vector, fx::Vector, gx)
        DiffEqDiffTools.finite_difference_jacobian!(gx, f!, x, jac_cache)
    end
    g!(x::Vector, gx::Array)  = (fx = similar(x); fg!(x, fx, gx))
    NLsolve.DifferentiableMultivariateFunction(f!.loss, g!, fg!)
end
