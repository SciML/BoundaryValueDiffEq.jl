@inline __default_cost(::Nothing) = (x, p) -> 0.0
@inline __default_cost(f) = f
@inline __build_cost(::Nothing, cache, mesh, M; kwargs...) = (x, p) -> 0.0
@inline function __build_cost(fun, cache, mesh, M; fit_parameters = false, p = nothing)
    if fit_parameters && SciMLStructures.isscimlstructure(p)
        # When fit_parameters=true, the state vector is augmented with tunable params
        # Extract them and use SciMLStructures.replace to update p for the cost function
        tunable_part, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
        l_params = length(tunable_part)
        length_u = M - l_params
        cost_fun = @views function (u, p_orig)
            newy = [u[i:(i + M - 1)] for i in 1:M:(length(u) - M + 1)]
            # Extract tunable params from first mesh point (same at all points)
            params_from_u = u[(length_u + 1):M]
            new_p = SciMLStructures.replace(SciMLStructures.Tunable(), p_orig, params_from_u)
            eval_sol = EvalSol(newy, mesh, cache)
            return fun(eval_sol, new_p)
        end
    elseif fit_parameters && !isnothing(p)
        length_u = M - length(p)
        cost_fun = @views function (u, p)
            # When fit_parameters=true, the state vector is augmented with tunable params
            newy = [u[i:(i + M - 1)] for i in 1:M:(length(u) - M + 1)]
            params_from_u = u[(length_u + 1):M]
            eval_sol = EvalSol(newy, mesh, cache)
            return fun(eval_sol, params_from_u)
        end
    else
        cost_fun = @views function (u, p)
            # simple recursive unflatten
            newy = [u[i:(i + M - 1)] for i in 1:M:(length(u) - M + 1)]
            eval_sol = EvalSol(newy, mesh, cache)
            return fun(eval_sol, p)
        end
    end
    return cost_fun
end

@inline function __extract_lcons_ucons(
        prob::AbstractBVProblem, ::Type{T}, M, N, bcresid_prototype, f_prototype
    ) where {T}
    L_f_prototype = length(f_prototype)
    L_bcresid_prototype = length(bcresid_prototype)
    lcons = if isnothing(prob.lcons)
        zeros(T, L_bcresid_prototype + (N - 1) * L_f_prototype)
    else
        lcons_length = length(prob.lcons)
        vcat(prob.lcons, zeros(T, N * M - lcons_length))
    end
    ucons = if isnothing(prob.ucons)
        zeros(T, L_bcresid_prototype + (N - 1) * L_f_prototype)
    else
        ucons_length = length(prob.ucons)
        vcat(prob.ucons, zeros(T, N * M - ucons_length))
    end
    return lcons, ucons
end

@inline function __extract_lcons_ucons(
        prob::AbstractBVProblem, ::Type{T}, M, N, bcresid_prototype, ::Nothing
    ) where {T}
    lcons = zeros(T, N * M)
    ucons = zeros(T, N * M)
    return lcons, ucons
end

@inline function __extract_lcons_ucons(prob::AbstractBVProblem, ::Type{T}, M, N) where {T}
    lcons = if isnothing(prob.lcons)
        zeros(T, N * M)
    else
        lcons_length = length(prob.lcons)
        vcat(prob.lcons, zeros(T, N * M - lcons_length))
    end
    ucons = if isnothing(prob.ucons)
        zeros(T, N * M)
    else
        ucons_length = length(prob.ucons)
        vcat(prob.ucons, zeros(T, N * M - ucons_length))
    end
    return lcons, ucons
end

@inline function __extract_lb_ub(prob::AbstractBVProblem, ::Type{T}, M, N) where {T}
    lb = if isnothing(prob.lb)
        nothing
    else
        repeat(prob.lb, N)
    end
    ub = if isnothing(prob.ub)
        nothing
    else
        repeat(prob.ub, N)
    end
    return lb, ub
end

function integral(fun, domain)
    prob = IntegralProblem(fun, domain)
    sol = SciMLBase.solve(prob, Integrals.QuadGKJL())
    return sol.u
end

"""
    __construct_internal_problem

Constructs the internal problem according to the specified boundary value problem and the
selected algorithm. Depending on the formulation, it returns either a `NonlinearProblem` or an `OptimizationProblem`.
"""
function __construct_internal_problem(
        prob, pt::StandardBVProblem, alg, loss, jac, jac_prototype, resid_prototype,
        bcresid_prototype, f_prototype, y, p, M::Int, N::Int, cost_fun
    )
    T = eltype(y)
    iip = SciMLBase.isinplace(prob)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{iip}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{true}(
            cost_fun,
            AutoSparse(
                get_dense_ad(alg.jac_alg.nonbc_diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N, bcresid_prototype, f_prototype)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end

function __construct_internal_problem(
        prob, pt::TwoPointBVProblem, alg, loss, jac, jac_prototype, resid_prototype,
        bcresid_prototype, f_prototype, y, p, M::Int, N::Int, cost_fun
    )
    T = eltype(y)
    iip = SciMLBase.isinplace(prob)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{iip}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{true}(
            cost_fun,
            AutoSparse(
                get_dense_ad(alg.jac_alg.diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N, bcresid_prototype, f_prototype)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end

# Single shooting use diffmode for StandardBVProblem and TwoPointBVProblem
# Version with Val{iip} for type stability
function __construct_internal_problem(
        prob::SciMLBase.AbstractBVProblem, alg, loss, jac,
        jac_prototype, resid_prototype, y, p, M::Int, N::Int, ::Nothing, ::Val{iip}
    ) where {iip}
    T = eltype(y)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{iip}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{iip}(
            __default_cost(prob.f.cost),
            AutoSparse(
                get_dense_ad(alg.jac_alg.diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end

# Fallback version without Val{iip} - extracts iip at runtime (for backwards compatibility)
function __construct_internal_problem(
        prob::SciMLBase.AbstractBVProblem, alg, loss, jac,
        jac_prototype, resid_prototype, y, p, M::Int, N::Int, ::Nothing
    )
    return __construct_internal_problem(
        prob, alg, loss, jac, jac_prototype, resid_prototype,
        y, p, M, N, nothing, Val(SciMLBase.isinplace(prob))
    )
end

# Multiple shooting always use inplace version internal problem constructor
function __construct_internal_problem(
        prob, pt::StandardBVProblem, alg, loss, jac, jac_prototype,
        resid_prototype, y, p, M::Int, N::Int, ::Nothing, ::Nothing
    )
    T = eltype(y)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{true}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{true}(
            __default_cost(prob.f.cost),
            AutoSparse(
                get_dense_ad(alg.jac_alg.nonbc_diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.nonbc_diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N, bcresid_prototype, f_prototype)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end
function __construct_internal_problem(
        prob, pt::TwoPointBVProblem, alg, loss, jac, jac_prototype,
        resid_prototype, y, p, M::Int, N::Int, ::Nothing, ::Nothing
    )
    T = eltype(y)
    #iip = SciMLBase.isinplace(prob)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{true}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{true}(
            __default_cost(prob.f.cost),
            AutoSparse(
                get_dense_ad(alg.jac_alg.diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.nonbc_diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N, bcresid_prototype, f_prototype)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end

# SecondOrderBVProblem
function __construct_internal_problem(
        prob, pt::StandardSecondOrderBVProblem, alg, loss, jac,
        jac_prototype, resid_prototype, y, p, M::Int, N::Int
    )
    T = eltype(y)
    iip = SciMLBase.isinplace(prob)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{iip}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{iip}(
            __default_cost(prob.f.cost),
            AutoSparse(
                get_dense_ad(alg.jac_alg.nonbc_diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.nonbc_diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end
function __construct_internal_problem(
        prob, pt::TwoPointSecondOrderBVProblem, alg, loss, jac,
        jac_prototype, resid_prototype, y, p, M::Int, N::Int
    )
    T = eltype(y)
    iip = SciMLBase.isinplace(prob)
    if !isnothing(alg.nlsolve) || (isnothing(alg.nlsolve) && isnothing(alg.optimize))
        nlf = NonlinearFunction{iip}(
            loss; jac = jac, resid_prototype = resid_prototype,
            jac_prototype = jac_prototype
        )
        return __internal_nlsolve_problem(prob, resid_prototype, y, nlf, y, p)
    else
        optf = OptimizationFunction{true}(
            __default_cost(prob.f),
            AutoSparse(
                get_dense_ad(alg.jac_alg.diffmode),
                sparsity_detector = __default_sparsity_detector(alg.jac_alg.nonbc_diffmode)
            ),
            cons = loss,
            cons_j = jac,
            cons_jac_prototype = sparse(jac_prototype)
        )
        lcons, ucons = __extract_lcons_ucons(prob, T, M, N)
        lb, ub = __extract_lb_ub(prob, T, M, N)

        return __internal_optimization_problem(
            prob, optf, y, p; lcons = lcons, ucons = ucons, lb = lb, ub = ub
        )
    end
end
