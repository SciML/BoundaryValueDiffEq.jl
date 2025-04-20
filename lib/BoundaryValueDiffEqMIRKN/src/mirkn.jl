@concrete mutable struct MIRKNCache{iip, T} <: AbstractBoundaryValueDiffEqCache
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

function SciMLBase.__init(prob::SecondOrderBVProblem, alg::AbstractMIRKN; dt = 0.0,
        adaptive = false, controller = NoController(), kwargs...)
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
    k_discrete = [__maybe_allocate_diffcache(
                      safe_similar(X, M, stage), chunksize, alg.jac_alg) for _ in 1:Nig]

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
        alg_order(alg), stage, M, size(X), f, bc, prob_, prob.problem_type, prob.p, alg,
        TU, bcresid_prototype, mesh, mesh_dt, k_discrete, y, y₀, residual, fᵢ_cache,
        fᵢ₂_cache, resid_size, (; abstol, dt, adaptive, controller, kwargs...))
end

function SciMLBase.solve!(cache::MIRKNCache{iip, T}) where {iip, T}
    (_, _, _), kwargs = __split_kwargs(; cache.kwargs...)
    info::ReturnCode.T = ReturnCode.Success

    sol_nlprob, info = __perform_mirkn_iteration(cache; kwargs...)

    solu = ArrayPartition.(
        cache.y₀.u[1:length(cache.mesh)], cache.y₀.u[(length(cache.mesh) + 1):end])
    odesol = SciMLBase.build_solution(
        cache.prob, cache.alg, cache.mesh, solu; retcode = info)
    return __build_solution(cache.prob, odesol, sol_nlprob)
end

function __perform_mirkn_iteration(cache::MIRKNCache; nlsolve_kwargs = (;), kwargs...)
    nlprob::NonlinearProblem = __construct_nlproblem(cache, vec(cache.y₀), copy(cache.y₀))
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, cache.alg.nlsolve)
    sol_nlprob = __solve(nlprob, nlsolve_alg; kwargs..., nlsolve_kwargs..., alias_u0 = true)
    recursive_unflatten!(cache.y₀, sol_nlprob.u)

    return sol_nlprob, sol_nlprob.retcode
end

# Constructing the Nonlinear Problem
function __construct_nlproblem(
        cache::MIRKNCache{iip}, y::AbstractVector, y₀::AbstractVectorOfArray) where {iip}
    pt = cache.problem_type

    eval_sol = EvalSol(
        __restructure_sol(y₀.u[1:length(cache.mesh)], cache.in_size), cache.mesh, cache)
    eval_dsol = EvalSol(
        __restructure_sol(y₀.u[(length(cache.mesh) + 1):end], cache.in_size),
        cache.mesh, cache)

    loss_bc = if iip
        @closure (du, u, p) -> __mirkn_loss_bc!(
            du, u, p, pt, cache.bc, cache.y, cache.mesh, cache)
    else
        @closure (u, p) -> __mirkn_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh, cache)
    end

    loss_collocation = if iip
        @closure (du, u, p) -> __mirkn_loss_collocation!(
            du, u, p, cache.y, cache.mesh, cache.residual, cache)
    else
        @closure (u, p) -> __mirkn_loss_collocation(
            u, p, cache.y, cache.mesh, cache.residual, cache)
    end

    loss = if iip
        @closure (du, u, p) -> __mirkn_loss!(
            du, u, p, cache.y, pt, cache.bc, cache.residual,
            cache.mesh, cache, eval_sol, eval_dsol)
    else
        @closure (u, p) -> __mirkn_loss(
            u, p, cache.y, pt, cache.bc, cache.mesh, cache, eval_sol, eval_dsol)
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss, pt)
end

function __construct_nlproblem(cache::MIRKNCache{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::StandardSecondOrderBVProblem) where {iip, BC, C, LF}
    (; jac_alg) = cache.alg
    N = length(cache.mesh)

    resid_bc = cache.bcresid_prototype
    L = length(resid_bc)
    resid_collocation = safe_similar(y, cache.M * (2 * N - 2))

    bc_diffmode = if jac_alg.bc_diffmode isa AutoSparse
        AutoSparse(get_dense_ad(jac_alg.bc_diffmode);
            sparsity_detector = __default_sparsity_detector(jac_alg.bc_diffmode),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.bc_diffmode))
    else
        jac_alg.bc_diffmode
    end

    cache_bc = if iip
        DI.prepare_jacobian(loss_bc, resid_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss_bc, bc_diffmode, y, Constant(cache.p))
    end

    nonbc_diffmode = if jac_alg.nonbc_diffmode isa AutoSparse
        AutoSparse(get_dense_ad(jac_alg.nonbc_diffmode);
            sparsity_detector = __default_sparsity_detector(jac_alg.nonbc_diffmode),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.nonbc_diffmode))
    else
        jac_alg.nonbc_diffmode
    end

    cache_collocation = if iip
        DI.prepare_jacobian(
            loss_collocation, resid_collocation, nonbc_diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss_collocation, nonbc_diffmode, y, Constant(cache.p))
    end

    J_bc = if iip
        DI.jacobian(loss_bc, resid_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss_bc, cache_bc, bc_diffmode, y, Constant(cache.p))
    end
    J_c = if iip
        DI.jacobian(loss_collocation, resid_collocation, cache_collocation,
            nonbc_diffmode, y, Constant(cache.p))
    else
        DI.jacobian(
            loss_collocation, cache_collocation, nonbc_diffmode, y, Constant(cache.p))
    end

    jac_prototype = vcat(J_bc, J_c)

    jac = if iip
        @closure (J, u, p) -> __mirkn_mpoint_jacobian!(
            J, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc, cache_collocation,
            loss_bc, loss_collocation, resid_bc, resid_collocation, L, cache.p)
    else
        @closure (u, p) -> __mirkn_mpoint_jacobian(
            jac_prototype, J_c, u, bc_diffmode, nonbc_diffmode, cache_bc,
            cache_collocation, loss_bc, loss_collocation, L, cache.p)
    end
    resid_prototype = vcat(resid_bc, resid_collocation)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)
    __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __construct_nlproblem(cache::MIRKNCache{iip}, y, loss_bc::BC, loss_collocation::C,
        loss::LF, ::TwoPointSecondOrderBVProblem) where {iip, BC, C, LF}
    (; nlsolve, jac_alg) = cache.alg
    N = length(cache.mesh)

    resid = vcat(@view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
        safe_similar(y, cache.M * 2 * (N - 1)),
        @view(cache.bcresid_prototype[(prod(cache.resid_size[1]) + 1):end]))

    diffmode = if jac_alg.diffmode isa AutoSparse
        AutoSparse(get_dense_ad(jac_alg.diffmode);
            sparsity_detector = __default_sparsity_detector(jac_alg.diffmode),
            coloring_algorithm = __default_coloring_algorithm(jac_alg.diffmode))
    else
        jac_alg.diffmode
    end

    diffcache = if iip
        DI.prepare_jacobian(loss, resid, diffmode, y, Constant(cache.p))
    else
        DI.prepare_jacobian(loss, diffmode, y, Constant(cache.p))
    end

    jac_prototype = if iip
        DI.jacobian(loss, resid, diffcache, diffmode, y, Constant(cache.p))
    else
        DI.jacobian(loss, diffcache, diffmode, y, Constant(cache.p))
    end

    jac = if iip
        @closure (J, u, p) -> __mirkn_2point_jacobian!(
            J, u, jac_alg.diffmode, diffcache, loss, resid, p)
    else
        @closure (u, p) -> __mirkn_2point_jacobian(
            u, jac_prototype, jac_alg.diffmode, diffcache, loss, p)
    end

    resid_prototype = copy(resid)
    nlf = NonlinearFunction{iip}(
        loss; jac = jac, resid_prototype = resid_prototype, jac_prototype = jac_prototype)
    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __mirkn_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid, p) where {L}
    DI.jacobian!(loss_fn, resid, J, diffcache, diffmode, x, Constant(p))
    return J
end

function __mirkn_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L, p) where {L}
    DI.jacobian!(loss_fn, J, diffcache, diffmode, x, Constant(p))
    return J
end

function __mirkn_mpoint_jacobian!(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache, loss_bc::BC,
        loss_collocation::C, resid_bc, resid_collocation, L::Int, p) where {BC, C}
    DI.jacobian!(
        loss_bc, resid_bc, @view(J[1:L, :]), bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, resid_collocation, @view(J[(L + 1):end, :]),
        nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    return nothing
end

function __mirkn_mpoint_jacobian(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache,
        loss_bc::BC, loss_collocation::C, L::Int, p) where {BC, C}
    DI.jacobian!(loss_bc, @view(J[1:L, :]), bc_diffcache, bc_diffmode, x, Constant(p))
    DI.jacobian!(loss_collocation, @view(J[(L + 1):end, :]),
        nonbc_diffcache, nonbc_diffmode, x, Constant(p))
    return J
end

@views function __mirkn_loss!(resid, u, p, y, pt::StandardSecondOrderBVProblem, bc::BC,
        residual, mesh, cache::MIRKNCache, EvalSol, EvalDSol) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[3:end], cache, y_, u, p)
    EvalSol.u[1:end] .= __restructure_sol(y_[1:length(cache.mesh)], cache.in_size)
    EvalSol.cache.k_discrete[1:end] .= cache.k_discrete
    EvalDSol.u[1:end] .= __restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size)
    EvalDSol.cache.k_discrete[1:end] .= cache.k_discrete
    eval_bc_residual!(resids[1:2], pt, bc, EvalSol, EvalDSol, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(u, p, y, pt::StandardSecondOrderBVProblem, bc::BC,
        mesh, cache::MIRKNCache, EvalSol, EvalDSol) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, p)
    EvalSol.u[1:end] .= __restructure_sol(y_[1:length(cache.mesh)], cache.in_size)
    EvalSol.cache.k_discrete[1:end] .= cache.k_discrete
    EvalDSol.u[1:end] .= __restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size)
    EvalDSol.cache.k_discrete[1:end] .= cache.k_discrete
    resid_bc = eval_bc_residual(pt, bc, EvalSol, EvalDSol, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __mirkn_loss_bc!(
        resid, u, p, pt, bc!::BC, y, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = EvalSol(__restructure_sol(y_[1:length(cache.mesh)], cache.in_size), mesh, cache)
    dsoly_ = EvalSol(__restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size),
        cache.mesh, cache)
    eval_bc_residual!(resid, pt, bc!, soly_, dsoly_, p, mesh)
    return nothing
end

@views function __mirkn_loss_bc(u, p, pt, bc!::BC, y, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = EvalSol(__restructure_sol(y_[1:length(cache.mesh)], cache.in_size), mesh, cache)
    dsoly_ = EvalSol(__restructure_sol(y_[(length(cache.mesh) + 1):end], cache.in_size),
        cache.mesh, cache)
    return eval_bc_residual(pt, bc!, soly_, dsoly_, p, mesh)
end

@views function __mirkn_loss_collocation!(resid, u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual[3:end]]
    Φ!(resids, cache, y_, u, p)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss_collocation(u, p, y, mesh, residual, cache)
    y_ = recursive_unflatten!(y, u)
    resids = Φ(cache, y_, u, p)
    return mapreduce(vec, vcat, resids)
end

@views function __mirkn_loss!(resid, u, p, y, pt::TwoPointSecondOrderBVProblem, bc!::BC,
        residual, mesh, cache::MIRKNCache, _, _) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    soly_ = VectorOfArray(y_)
    Φ!(resids[3:end], cache, y_, u, p)
    eval_bc_residual!(resids[1:2], pt, bc!, soly_, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(u, p, y, pt::TwoPointSecondOrderBVProblem,
        bc!::BC, mesh, cache::MIRKNCache, _, _) where {BC}
    y_ = recursive_unflatten!(y, u)
    soly_ = VectorOfArray(y_)
    resid_co = Φ(cache, y_, u, p)
    resid_bc = eval_bc_residual(pt, bc!, soly_, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end
