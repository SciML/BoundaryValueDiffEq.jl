function SciMLBase.__solve(
        prob::BVProblem, alg_::Shooting; abstol = 1.0e-6,
        odesolve_kwargs = (;), nlsolve_kwargs = (; abstol = abstol),
        optimize_kwargs = (; abstol = abstol), verbose = true, kwargs...
    )
    # Setup the problem
    if prob.u0 isa AbstractArray{<:Number}
        u0 = prob.u0
    else
        verbose && @warn "Initial guess provided, but will be ignored for Shooting."
        u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    end

    alg = concretize_jacobian_algorithm(alg_, prob)
    (; diffmode) = alg.jac_alg

    bcresid_prototype, resid_size = __get_bcresid_prototype(prob, u0)
    iip, bc, u0, u0_size = isinplace(prob), prob.f.bc, deepcopy(u0), size(u0)
    @assert (iip || isnothing(alg_.optimize)) "Out-of-place constraints don't allow optimization solvers "
    resid_prototype = __vec(bcresid_prototype)

    # Construct the residual function
    actual_ode_kwargs = (; kwargs..., verbose, odesolve_kwargs...)
    # For TwoPointBVPs we don't need to save every step
    if prob.problem_type isa TwoPointBVProblem
        ode_kwargs = (; save_everystep = false, actual_ode_kwargs...)
    else
        ode_kwargs = (; actual_ode_kwargs...)
    end
    internal_prob = ODEProblem{iip}(prob.f, u0, prob.tspan, prob.p)
    ode_cache_loss_fn = SciMLBase.__init(internal_prob, alg.ode_alg; ode_kwargs...)

    loss_fn = if iip
        @closure (
            du,
            u,
            p,
        ) -> __single_shooting_loss!(
            du, u, p, ode_cache_loss_fn, bc, u0_size, prob.problem_type, resid_size
        )
    else
        @closure (
            u,
            p,
        ) -> __single_shooting_loss(
            u, p, ode_cache_loss_fn, bc, u0_size, prob.problem_type
        )
    end

    y_ = similar(resid_prototype)

    jac_cache = if iip
        DI.prepare_jacobian(nothing, y_, diffmode, __vec(u0); strict = Val(false))
    else
        DI.prepare_jacobian(nothing, diffmode, __vec(u0); strict = Val(false))
    end

    ode_cache_jac_fn = __single_shooting_jacobian_ode_cache(
        internal_prob, jac_cache, __cache_trait(diffmode),
        diffmode, u0, alg.ode_alg; ode_kwargs...
    )

    loss_fnₚ = if iip
        @closure (
            du,
            u,
        ) -> __single_shooting_loss!(
            du, u, prob.p, ode_cache_jac_fn, bc, u0_size, prob.problem_type, resid_size
        )
    else
        @closure (u) -> __single_shooting_loss(
            u, prob.p, ode_cache_jac_fn, bc, u0_size, prob.problem_type
        )
    end

    jac_prototype = if iip
        DI.jacobian(loss_fnₚ, y_, jac_cache, diffmode, __vec(u0))
    else
        DI.jacobian(loss_fnₚ, jac_cache, diffmode, __vec(u0))
    end

    jac_fn = if iip
        @closure (
            J, u, p,
        ) -> __single_shooting_jacobian!(J, u, jac_cache, diffmode, loss_fnₚ, y_)
    else
        @closure (
            u,
            p,
        ) -> __single_shooting_jacobian(
            jac_prototype, u, jac_cache, diffmode, loss_fnₚ
        )
    end

    nlprob = __construct_internal_problem(
        prob, alg, loss_fn, jac_fn, jac_prototype,
        resid_prototype, u0, prob.p, length(u0), 1, nothing
    )
    solve_alg = __concrete_solve_algorithm(nlprob, alg.nlsolve, alg.optimize)
    kwargs = __concrete_kwargs(alg.nlsolve, alg.optimize, nlsolve_kwargs, optimize_kwargs)
    nlsol = __internal_solve(nlprob, solve_alg; kwargs...)

    # There is no way to reinit with the same cache with different cache. But not saving
    # the internal values gives a significant speedup. So we just create a new cache
    internal_prob_final = ODEProblem{iip}(prob.f, reshape(nlsol.u, u0_size), prob.tspan, prob.p)
    odesol = __solve(internal_prob_final, alg.ode_alg; actual_ode_kwargs...)

    return __build_solution(prob, odesol, nlsol)
end

function __single_shooting_loss!(
        resid_, u0_, p, cache, bc::BC, u0_size,
        pt::TwoPointBVProblem, (resida_size, residb_size)
    ) where {BC}
    resida = @view resid_[1:prod(resida_size)]
    residb = @view resid_[(prod(resida_size) + 1):end]
    resid = (reshape(resida, resida_size), reshape(residb, residb_size))

    SciMLBase.reinit!(cache, reshape(u0_, u0_size))
    odesol = solve!(cache)

    eval_bc_residual!(resid, pt, bc, odesol, p)

    return nothing
end

function __single_shooting_loss!(
        resid_, u0_, p, cache, bc::BC, u0_size,
        pt::StandardBVProblem, resid_size
    ) where {BC}
    resid = reshape(resid_, resid_size)

    SciMLBase.reinit!(cache, reshape(u0_, u0_size))
    odesol = solve!(cache)

    eval_bc_residual!(resid, pt, bc, odesol, p)

    return nothing
end

function __single_shooting_loss(u, p, cache, bc::BC, u0_size, pt) where {BC}
    SciMLBase.reinit!(cache, reshape(u, u0_size))
    odesol = solve!(cache)
    return __vec(eval_bc_residual(pt, bc, odesol, p))
end

function __single_shooting_jacobian!(J, u, jac_cache, diffmode, loss_fn::L, fu) where {L}
    DI.jacobian!(loss_fn, fu, J, jac_cache, diffmode, __vec(u))
    return J
end

function __single_shooting_jacobian(J, u, jac_cache, diffmode, loss_fn::L) where {L}
    DI.jacobian!(loss_fn, J, jac_cache, diffmode, __vec(u))
    return J
end

function __single_shooting_jacobian_ode_cache(
        prob, jac_cache, ::NoDiffCacheNeeded, diffmode, u0, ode_alg; kwargs...
    )
    return SciMLBase.__init(remake(prob; u0), ode_alg; kwargs...)
end

function __single_shooting_jacobian_ode_cache(
        prob, jac_cache, ::DiffCacheNeeded, diffmode, u0, ode_alg; kwargs...
    )
    T_dual = eltype(overloaded_input_type(jac_cache))
    xduals = zeros(T_dual, size(u0))
    prob_ = remake(prob; u0 = reshape(xduals, size(u0)), tspan = eltype(xduals).(prob.tspan))
    return SciMLBase.__init(prob_, ode_alg; kwargs...)
end
