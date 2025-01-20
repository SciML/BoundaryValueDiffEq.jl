@concrete mutable struct MIRKNCache{iip, T}
    order::Int                 # The order of MIRKN method
    stage::Int                 # The state of MIRKN method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # SecondOrderBVProblem
    problem_type               # StandardSecondOrderBVProblem
    p                          # Parameters
    alg                        # MIRKN methods
    TU                         # MIRKN Tableau
    bcresid_prototype
    mesh                       # Discrete mesh
    mesh_dt
    k_discrete                 # Stage information associated with the discrete Runge-Kutta-Nyström method
    y
    y₀
    residual
    fᵢ_cache
    fᵢ₂_cache
    resid_size
    kwargs
end

Base.eltype(::MIRKNCache{iip, T}) where {iip, T} = T

function SciMLBase.__init(prob::SecondOrderBVProblem, alg::AbstractMIRKN;
        dt = 0.0, adaptive = false, kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    iip = isinplace(prob)
    t₀, t₁ = prob.tspan
    ig, T, M, Nig, X = __extract_problem_details(prob; dt, check_positive_dt = true)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)

    TU = constructMIRKN(alg, T)

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_guess_on_mesh(prob, prob.u0, Nig, prob.p, false)
    chunksize = pickchunksize(M * (2 * Nig - 2))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(zero(x)), chunksize, alg.jac_alg)

    y = __alloc.(copy.(y₀.u))
    fᵢ_cache = __alloc(zero(X))
    fᵢ₂_cache = __alloc(zero(X))
    stage = alg_stage(alg)
    bcresid_prototype = zero(vcat(X, X))
    k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
                  for _ in 1:Nig]

    residual = if iip
        __alloc.(copy.(@view(y₀.u[1:end])))
    else
        nothing
    end

    resid_size = size(bcresid_prototype)
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = @closure (ddu, du, u, p, t) -> __vec_f!(ddu, du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointSecondOrderBVProblem)
            @closure (r, du, u, p, t) -> __vec_so_bc!(
                r, du, u, p, t, prob.f.bc, resid_size, size(X))
        else
            (
                @closure((r, du, u, p)->__vec_so_bc!(
                    r, du, u, p, first(prob.f.bc), resid_size[1], size(X))),
                @closure((r, du, u, p)->__vec_so_bc!(
                    r, du, u, p, last(prob.f.bc), resid_size[2], size(X))))
        end
        vecf!, vecbc!
    else
        vecf = @closure (du, u, p, t) -> __vec_f(du, u, p, t, prob.f, size(X))
        vecbc = if !(prob.problem_type isa TwoPointSecondOrderBVProblem)
            @closure (du, u, p, t) -> __vec_so_bc(du, u, p, t, prob.f.bc, size(X))
        else
            (@closure((du, u, p)->__vec_so_bc(du, u, p, first(prob.f.bc), size(X))),
                @closure((du, u, p)->__vec_so_bc(du, u, p, last(prob.f.bc), size(X))))
        end
        vecf, vecbc
    end

    prob_ = !(prob.u0 isa AbstractArray) ? remake(prob; u0 = X) : prob

    return MIRKNCache{iip, T}(
        alg_order(alg), stage, M, size(X), f, bc, prob_, prob.problem_type,
        prob.p, alg, TU, bcresid_prototype, mesh, mesh_dt, k_discrete, y,
        y₀, residual, fᵢ_cache, fᵢ₂_cache, resid_size, (; dt, kwargs...))
end

function __split_mirkn_kwargs(; dt, kwargs...)
    return ((dt), (; kwargs...))
end

function SciMLBase.solve!(cache::MIRKNCache{iip, T}) where {iip, T}
    (_), kwargs = __split_mirkn_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success

    sol_nlprob, info = __perform_mirkn_iteration(cache; kwargs...)

    solu = ArrayPartition.(
        cache.y₀.u[1:length(cache.mesh)], cache.y₀.u[(length(cache.mesh) + 1):end])
    odesol = SciMLBase.build_solution(
        cache.prob, cache.alg, cache.mesh, solu; retcode = info)
    return __build_solution(cache.prob, odesol, sol_nlprob)
end

function __perform_mirkn_iteration(cache::MIRKNCache; nlsolve_kwargs = (;), kwargs...)
    nlprob::NonlinearProblem = __construct_nlproblem(cache, vec(cache.y₀))
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, cache.alg.nlsolve)
    sol_nlprob = __solve(nlprob, nlsolve_alg; kwargs..., nlsolve_kwargs..., alias_u0 = true)
    recursive_unflatten!(cache.y₀, sol_nlprob.u)

    return sol_nlprob, sol_nlprob.retcode
end

function __construct_nlproblem(cache::MIRKNCache{iip}, y::AbstractVector) where {iip}
    (; alg) = cache
    pt = cache.problem_type
    loss = if iip
        @closure (du, u, p) -> __mirkn_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual, cache.mesh, cache)
    else
        @closure (u, p) -> __mirkn_loss(u, p, cache.y, pt, cache.bc, cache.mesh, cache)
    end

    lossₚ = (iip ? __Fix3 : Base.Fix2)(loss, cache.p)
    sd = alg.jac_alg.diffmode isa AutoSparse ? SymbolicsSparsityDetection() :
         NoSparsityDetection()
    ad = alg.jac_alg.diffmode
    lz = __similar(y)
    jac_cache = __sparse_jacobian_cache(Val(iip), ad, sd, lossₚ, lz, y)
    jac_prototype = init_jacobian(jac_cache)
    jac = if iip
        @closure (J, u, p) -> __mirkn_mpoint_jacobian!(J, u, ad, jac_cache, lossₚ, lz)
    else
        @closure (u, p) -> __mirkn_mpoint_jacobian(jac_prototype, u, ad, jac_cache, lossₚ)
    end
    resid_prototype = zero(lz)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)
    return __internal_nlsolve_problem(cache.prob, resid_prototype, lz, nlf, lz, cache.p)
end

function __mirkn_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, resid, x)
    return J
end

function __mirkn_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, x)
    return J
end

function __mirkn_mpoint_jacobian!(J, x, diffmode, diffcache, loss, resid)
    sparse_jacobian!(J, diffmode, diffcache, loss, resid, x)
    return nothing
end

function __mirkn_mpoint_jacobian(J, x, diffmode, diffcache, loss)
    sparse_jacobian!(J, diffmode, diffcache, loss, x)
    return J
end

@views function __mirkn_loss!(resid, u, p, y, pt::StandardSecondOrderBVProblem,
        bc::BC, residual, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[3:end], cache, y_, u, p)
    soly_ = EvalSol(
        __restructure_sol(y_[1:length(cache.mesh)], cache.in_size), cache.mesh, cache)
    dsoly_ = EvalSol(__restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size),
        cache.mesh, cache)
    eval_bc_residual!(resids[1:2], pt, bc, soly_, dsoly_, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(u, p, y, pt::StandardSecondOrderBVProblem,
        bc::BC, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, p)
    soly_ = EvalSol(
        __restructure_sol(y_[1:length(cache.mesh)], cache.in_size), cache.mesh, cache)
    dsoly_ = EvalSol(__restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size),
        cache.mesh, cache)
    resid_bc = eval_bc_residual(pt, bc, soly_, dsoly_, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __mirkn_loss!(resid, u, p, y, pt::TwoPointSecondOrderBVProblem,
        bc!::BC, residual, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[3:end], cache, y_, u, p)
    eval_bc_residual!(resids, pt, bc!, soly_, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(u, p, y, pt::TwoPointSecondOrderBVProblem,
        bc!::BC, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resid_co = Φ(cache, y_, u, p)
    resid_bc = eval_bc_residual(pt, bc!, soly_, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end
