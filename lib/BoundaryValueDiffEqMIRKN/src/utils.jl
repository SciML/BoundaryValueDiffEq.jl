function general_eval_bc_residual(
        ::StandardSecondOrderBVProblem, bc::BC, y, p, mesh) where {BC}
    M = length(y[1])
    L = length(mesh)
    res_bc = bc(y[(L + 1):end], y[1:L], p, mesh)
    return res_bc
end
function eval_bc_residual(::StandardSecondOrderBVProblem, bc::BC, y, dy, p, mesh) where {BC}
    res_bc = bc(dy, y, p, mesh)
    return res_bc
end
function eval_bc_residual(::TwoPointSecondOrderBVProblem, bc::BC, sol, p, mesh) where {BC}
    M = length(sol[1])
    L = length(mesh)
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))[1:M]
    ub = sol isa AbstractVector ? sol[L] : sol(last(t))[1:M]
    dua = sol isa AbstractVector ? sol[L + 1] : sol(first(t))[(M + 1):end]
    dub = sol isa AbstractVector ? sol[end] : sol(last(t))[(M + 1):end]
    return vcat(bc[1](dua, ua, p), bc[2](dub, ub, p))
end

function general_eval_bc_residual!(
        resid, ::StandardSecondOrderBVProblem, bc!::BC, y, p, mesh) where {BC}
    L = length(mesh)
    bc!(resid, y[(L + 1):end], y[1:L], p, mesh)
end
function eval_bc_residual!(
        resid, ::StandardSecondOrderBVProblem, bc!::BC, y, dy, p, mesh) where {BC}
    M = length(resid[1])
    res_bc = vcat(resid[1], resid[2])
    bc!(res_bc, dy, y, p, mesh)
    copyto!(resid[1], res_bc[1:M])
    copyto!(resid[2], res_bc[(M + 1):end])
end
function eval_bc_residual!(
        resid, ::TwoPointSecondOrderBVProblem, bc::BC, sol, p, mesh) where {BC}
    M = length(sol[1])
    L = length(mesh)
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))[1:M]
    ub = sol isa AbstractVector ? sol[L] : sol(last(t))[1:M]
    dua = sol isa AbstractVector ? sol[L + 1] : sol(first(t))[(M + 1):end]
    dub = sol isa AbstractVector ? sol[end] : sol(last(t))[(M + 1):end]
    bc[1](resid[1], dua, ua, p)
    bc[2](resid[2], dub, ub, p)
end

function __get_bcresid_prototype(::TwoPointSecondOrderBVProblem, prob::BVProblem, u)
    prototype = if prob.f.bcresid_prototype !== nothing
        prob.f.bcresid_prototype.x
    else
        first(prob.f.bc)(u, prob.p), last(prob.f.bc)(u, prob.p)
    end
    return prototype, size.(prototype)
end
function __get_bcresid_prototype(::StandardSecondOrderBVProblem, prob::BVProblem, u)
    prototype = prob.f.bcresid_prototype !== nothing ? prob.f.bcresid_prototype :
                __zeros_like(u)
    return prototype, size(prototype)
end

# Restructure Non-Vector Inputs
function __vec_f!(ddu, du, u, p, t, f!, u_size)
    f!(reshape(ddu, u_size), reshape(du, u_size), reshape(u, u_size), p, t)
    return nothing
end

__vec_f(du, u, p, t, f, u_size) = vec(f(reshape(du, u_size), reshape(u, u_size), p, t))

function __vec_so_bc!(resid, dsol, sol, p, t, bc!, resid_size, u_size)
    bc!(reshape(resid, resid_size), __restructure_sol(dsol, u_size),
        __restructure_sol(sol, u_size), p, t)
    return nothing
end

function __vec_so_bc!(resid, dsol, sol, p, bc!, resid_size, u_size)
    bc!(reshape(resid, resid_size), reshape(dsol, u_size), reshape(sol, u_size), p)
    return nothing
end

function __vec_so_bc(dsol, sol, p, t, bc, u_size)
    vec(bc(__restructure_sol(dsol, u_size), __restructure_sol(sol, u_size), p, t))
end
function __vec_so_bc(dsol, sol, p, bc, u_size)
    vec(bc(reshape(dsol, u_size), reshape(sol, u_size), p))
end

__get_non_sparse_ad(ad::AbstractADType) = ad

@inline function __initial_guess_on_mesh(
        prob::SecondOrderBVProblem, u₀::AbstractArray, Nig, p, alias_u0::Bool)
    return [copy(vec(u₀)) for _ in 1:(2 * (Nig + 1))]
end

function concrete_jacobian_algorithm(
        jac_alg::BVPJacobianAlgorithm, prob::AbstractBVProblem, alg)
    return concrete_jacobian_algorithm(jac_alg, prob.problem_type, prob, alg)
end

function concrete_jacobian_algorithm(
        jac_alg::BVPJacobianAlgorithm, prob_type, prob::SecondOrderBVProblem, alg)
    u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    diffmode = jac_alg.diffmode === nothing ? __default_sparse_ad(u0) : jac_alg.diffmode
    bc_diffmode = jac_alg.bc_diffmode === nothing ?
                  (prob_type isa TwoPointSecondOrderBVProblem ? __default_sparse_ad :
                   __default_nonsparse_ad)(u0) : jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? __default_sparse_ad(u0) :
                     jac_alg.nonbc_diffmode

    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end
