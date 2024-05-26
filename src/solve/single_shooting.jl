@concrete struct SingleShootingLossFunctionWrapper{iip} <: Function
    cache::OrdinaryDiffEq.ODEIntegrator # Don't specialize on the cache
    p
    bc
    u0_size
    problem_type
    resid_size
end

function ConstructionBase.constructorof(::Type{<:SingleShootingLossFunctionWrapper{iip}}) where {iip}
    return SingleShootingLossFunctionWrapper{iip}
end

@inline function (f::SingleShootingLossFunctionWrapper{true})(du, u, p = f.p)
    return __single_shooting_loss!(
        du, u, p, f.cache, f.bc, f.u0_size, f.problem_type, f.resid_size)
end
@inline function (f::SingleShootingLossFunctionWrapper{false})(u, p = f.p)
    return __single_shooting_loss(u, p, f.cache, f.bc, f.u0_size, f.problem_type)
end

function __solve(prob::BVProblem, alg_::Shooting; odesolve_kwargs = (;),
        nlsolve_kwargs = (;), verbose = true, kwargs...)
    # Setup the problem
    if prob.u0 isa AbstractArray{<:Number}
        u0 = prob.u0
    else
        verbose && @warn "Initial guess provided, but will be ignored for Shooting."
        u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    end
    T, N = eltype(u0), length(u0)

    alg = concretize_jacobian_algorithm(alg_, prob)

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)
    resid_prototype = __vec(bcresid_prototype)

    # Construct the residual function
    actual_ode_kwargs = (; kwargs..., verbose, odesolve_kwargs...)
    # For TwoPointBVPs we don't need to save every step
    ode_kwargs = merge(
        ifelse(prob.problem_type isa TwoPointBVProblem, (; save_everystep = false), (;)),
        actual_ode_kwargs)

    internal_prob = ODEProblem{iip}(prob.f, u0, prob.tspan, prob.p)
    ode_cache_loss = SciMLBase.__init(internal_prob, alg.ode_alg; ode_kwargs...)

    loss_fn = SingleShootingLossFunctionWrapper{iip}(
        ode_cache_loss, prob.p, bc, u0_size, prob.problem_type, resid_size)

    jac_extras = if iip
        DI.prepare_jacobian(
            loss_fn, similar(resid_prototype), alg.jac_alg.diffmode, vec(u0))
        # loss_fnₚ, similar(resid_prototype), alg.jac_alg.diffmode, vec(u0))
    else
        DI.prepare_jacobian(loss_fn, alg.jac_alg.diffmode, vec(u0))
        # DI.prepare_jacobian(loss_fnₚ, alg.jac_alg.diffmode, vec(u0))
    end

    ode_cache_jacobian = __single_shooting_jacobian_ode_cache(
        loss_fn, internal_prob, alg.jac_alg.diffmode, u0, alg.ode_alg; ode_kwargs...)
    loss_fn_jac = @set loss_fn.cache = ode_cache_jacobian

    @show jac_extras

    # DI doesn't have an interface to init the sparse jacobian
    jac_prototype = if iip
        DI.jacobian(
            loss_fnₚ, similar(resid_prototype), alg.jac_alg.diffmode, vec(u0), jac_extras)
    else
        DI.jacobian(loss_fnₚ, alg.jac_alg.diffmode, vec(u0), jac_extras)
    end

    # `p` is ignored in this function
    jac_fn = if iip
        @closure (J, u, p) -> begin
            @show typeof(J)
            resid = similar(resid_prototype,
                promote_type(eltype(u), eltype(p), eltype(resid_prototype)))
            DI.jacobian!(loss_fnₚ, resid, J, alg.jac_alg.diffmode, u, jac_extras)
        end
        #     @closure (J, u, p) -> __single_shooting_jacobian!(
        #         J, u, jac_cache, alg.jac_alg.diffmode, loss_fnₚ, y_)
    else
        J = jac_prototype
        @closure (u, p) -> begin
            DI.jacobian!(loss_fnₚ, J, alg.jac_alg.diffmode, u, jac_extras)
            return J
        end
        #     @closure (u, p) -> __single_shooting_jacobian(
        #         jac_prototype, u, jac_cache, alg.jac_alg.diffmode, loss_fnₚ)
    end

    @show typeof(jac_prototype)

    nlf = __unsafe_nonlinearfunction{iip}(
        loss_fn; jac_prototype, resid_prototype, jac = jac_fn)
    nlprob = __internal_nlsolve_problem(prob, resid_prototype, u0, nlf, vec(u0), prob.p)
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, alg.nlsolve)
    nlsol = __solve(nlprob, nlsolve_alg; nlsolve_kwargs..., verbose, kwargs...)

    # There is no way to reinit with the same cache with different cache. But not saving
    # the internal values gives a significant speedup. So we just create a new cache
    internal_prob_final = ODEProblem{iip}(
        prob.f, reshape(nlsol.u, u0_size), prob.tspan, prob.p)
    odesol = __solve(internal_prob_final, alg.ode_alg; actual_ode_kwargs...)

    return __build_solution(prob, odesol, nlsol)
end

function __single_shooting_loss!(resid_, u0_, p, cache, bc, u0_size, pt, resid_size)
    resid = if pt isa TwoPointBVProblem
        resida_size, residb_size = resid_size
        resida = @view resid_[1:prod(resida_size)]
        residb = @view resid_[(prod(resida_size) + 1):end]
        reshape(resida, resida_size), reshape(residb, residb_size)
    else
        reshape(resid_, resid_size)
    end

    reinit!(cache, reshape(u0_, u0_size))
    odesol = solve!(cache)
    eval_bc_residual!(resid, pt, bc, odesol, p)
end

function __single_shooting_loss(u, p, cache, bc, u0_size, pt)
    reinit!(cache, reshape(u, u0_size))
    odesol = solve!(cache)
    return __vec(eval_bc_residual(pt, bc, odesol, p))
end

function __single_shooting_jacobian_ode_cache(loss_fn, prob, ad, u0, ode_alg; kwargs...)
    if __cache_trait(ad) isa NoDiffCacheNeeded
        return SciMLBase.__init(remake(prob; u0), ode_alg; kwargs...)
    end

    ck, tag = __get_chunksize(ad), __get_tag(ad)
    tag = tag === nothing ? ForwardDiff.Tag{typeof(loss_fn), eltype(u0)} : tag
    x_partials = ForwardDiff.Partials{ck, eltype(u0)}.(tuple.(ntuple(Returns(u0), ck)...))
    x_duals = ForwardDiff.Dual{tag, eltype(u0), ck}.(u0, x_partials)
    prob = remake(prob; u0 = x_duals, tspan = eltype(x_duals).(prob.tspan))
    return SciMLBase.__init(prob, ode_alg; kwargs...)
end
