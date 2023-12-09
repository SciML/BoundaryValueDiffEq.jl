#= @concrete struct FIRKCache{iip, T} <: AbstractRKCache{iip, T}
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
    p                          # Parameters
    alg                        # FIRK methods
    TU                         # FIRK Tableau
    bcresid_prototype
    # Everything below gets resized in adaptive methods
    mesh                       # Discrete mesh
    mesh_dt                    # Step size
    k_discrete                 # Stage information associated with the discrete Runge-Kutta method
    y
    y₀
    residual
    # The following 2 caches are never resized
    fᵢ_cache
    fᵢ₂_cache
    defect
    kwargs
end
    # FIRK specific
    #nest_cache # cache for the nested nonlinear solve
    #p_nestprob =#

@concrete struct FIRKCache{iip, T} <: AbstractRKCache{iip, T}
    order::Int                 # The order of MIRK method
    stage::Int                 # The state of MIRK method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
    p                          # Parameters
    alg                        # MIRK methods
    TU                         # MIRK Tableau
    ITU                        # MIRK Interpolation Tableau
    bcresid_prototype
    # Everything below gets resized in adaptive methods
    mesh                       # Discrete mesh
    mesh_dt                    # Step size
    k_discrete                 # Stage information associated with the discrete Runge-Kutta method
    k_interp                   # Stage information associated with the discrete Runge-Kutta method
    y
    y₀
    residual
    # The following 2 caches are never resized
    fᵢ_cache
    fᵢ₂_cache
    defect
    p_nestprob
    nest_cache
    kwargs
end

function extend_y(y, N, stage)
    y_extended = similar(y, (N - 1) * (stage + 1) + 1)
    y_extended[1] = y[1]
    let ctr1 = 2
        for i in 2:N
            for j in 1:(stage + 1)
                y_extended[(ctr1)] = y[i]
                ctr1 += 1
            end
        end
    end
    return y_extended
end

function shrink_y(y, N, M, stage)
    y_shrink = similar(y, N)
    y_shrink[1] = y[1]
    let ctr = stage + 2
        for i in 2:N
            y_shrink[i] = y[ctr]
            ctr += (stage + 1)
        end
    end
    return y_shrink
end

function SciMLBase.__init(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
                          abstol = 1e-3, adaptive = true,
                          nlsolve_kwargs = (; abstol = 1e-3, reltol = 1e-3, maxiters = 10),
                          kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

    if adaptive && isa(alg, FIRKNoAdaptivity)
        error("Algorithm doesn't support adaptivity. Please choose a higher order algorithm.")
    end

    iip = isinplace(prob)
    has_initial_guess, T, M, n, X = __extract_problem_details(prob; dt,
                                                              check_positive_dt = true)
    stage = alg_stage(alg)
    TU, ITU = constructRK(alg, T)

    expanded_jac = isa(TU, FIRKTableau{false})
    chunksize = expanded_jac ? pickchunksize(M + M * n * (stage + 1)) :
                pickchunksize(M * (n + 1))

    __alloc_diffcache = x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    fᵢ_cache = __alloc_diffcache(similar(X))
    fᵢ₂_cache = vec(similar(X))

    # NOTE: Assumes the user provided initial guess is on a uniform mesh
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    mesh_dt = diff(mesh)

    defect_threshold = T(0.1)  # TODO: Allow user to specify these
    MxNsub = 3000              # TODO: Allow user to specify these

    # Don't flatten this here, since we need to expand it later if needed
    y₀ = expanded_jac ?
         extend_y(__initial_state_from_prob(prob, mesh), n + 1, alg_stage(alg)) :
         __initial_state_from_prob(prob, mesh)

    y = __alloc_diffcache.(copy.(y₀))

    k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
                  for _ in 1:n]
    k_interp = [similar(X, ifelse((adaptive && !isa(TU, FIRKTableau)), M, 0),
                        (adaptive && !isa(TU, FIRKTableau) ? ITU.s_star - stage : 0))
                for _ in 1:n]

    bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

    residual = if iip
        vcat([__alloc_diffcache(bcresid_prototype)],
             __alloc_diffcache.(copy.(@view(y₀[2:end]))))
    else
        nothing
    end

    defect = [similar(X, ifelse(adaptive, M, 0)) for _ in 1:n]

    # Transform the functions to handle non-vector inputs
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf!(du, u, p, t) = prob.f(reshape(du, size(X)), reshape(u, size(X)), p, t)
        vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
            function __vecbc!(resid, sol, p, t)
                prob.f.bc(reshape(resid, resid₁_size),
                          map(Base.Fix2(reshape, size(X)), sol), p, t)
            end
        else
            function __vecbc_a!(resida, ua, p)
                prob.f.bc[1](reshape(resida, resid₁_size[1]), reshape(ua, size(X)), p)
            end
            function __vecbc_b!(residb, ub, p)
                prob.f.bc[2](reshape(residb, resid₁_size[2]), reshape(ub, size(X)), p)
            end
            (__vecbc_a!, __vecbc_b!)
        end
        bcresid_prototype = vec(bcresid_prototype)
        vecf!, vecbc!
    else
        vecf(u, p, t) = vec(prob.f(reshape(u, size(X)), p, t))
        vecbc = if !(prob.problem_type isa TwoPointBVProblem)
            __vecbc(sol, p, t) = vec(prob.f.bc(map(Base.Fix2(reshape, size(X)), sol), p, t))
        else
            __vecbc_a(ua, p) = vec(prob.f.bc[1](reshape(ua, size(X)), p))
            __vecbc_b(ub, p) = vec(prob.f.bc[2](reshape(ub, size(X)), p))
            (__vecbc_a, __vecbc_b)
        end
        bcresid_prototype = vec(bcresid_prototype)
        vecf, vecbc
    end

    # Initialize internal nonlinear problem cache
    @unpack c, a, b, s = TU
    p_nestprob = zeros(T, M + 2)
    K0 = fill(one(T), (M, s))
    if iip
        nestprob = NonlinearProblem((res, K, p_nestprob) -> FIRK_nlsolve!(res, K,
                                                                          p_nestprob, f,
                                                                          a, c, stage,
                                                                          prob.p),
                                    K0, p_nestprob)
    else
        nlf = function (K, p_nestprob)
            res = zero(K)
            FIRK_nlsolve!(res, K, p_nestprob, f,
                          a, c, stage, prob.p)
            return res
        end
        nestprob = NonlinearProblem(nlf,
                                    K0, p_nestprob)
    end
    nest_cache = init(nestprob, NewtonRaphson(); nlsolve_kwargs...)

    return FIRKCache{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob,
                             prob.problem_type, prob.p, alg, TU, ITU, bcresid_prototype,
                             mesh,
                             mesh_dt,
                             k_discrete, k_interp, y, y₀, residual, fᵢ_cache, fᵢ₂_cache,
                             defect, p_nestprob, nest_cache,
                             (; defect_threshold, MxNsub, abstol, dt, adaptive, kwargs...))
end

"""
    __expand_cache!(cache::FIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function __expand_cache!(cache::FIRKCache)
    Nₙ = length(cache.mesh)
    __append_similar!(cache.k_discrete, Nₙ - 1, cache.M)
    __append_similar!(cache.y, Nₙ, cache.M, cache.TU)
    __append_similar!(cache.y₀, Nₙ, cache.M, cache.TU)
    __append_similar!(cache.residual, Nₙ, cache.M, cache.TU)
    __append_similar!(cache.defect, Nₙ - 1, cache.M)
    return cache
end

#= function solve_cache!(nest_cache, u, p_nest)
    reinit!(nest_cache, u, p = p_nest);
    return solve!(nest_cache)
end =#

function solve_cache!(nest_cache, p_nest)
    K = fill(one(eltype(nest_cache.u)), size(nest_cache.u))
    reinit!(nest_cache, K, p = p_nest)
    return solve!(nest_cache)
end

function _scalar_nlsolve_∂f_∂p(f, res, u, p)
    ff = p isa Number ? ForwardDiff.derivative :
         (u isa Number ? ForwardDiff.gradient : ForwardDiff.jacobian)
    return ff((y, x) -> f(y, u, x), res, p)
end

function _scalar_nlsolve_∂f_∂u(f, res, u, p)
    ff = u isa Number ? ForwardDiff.derivative : ForwardDiff.jacobian
    return ff((y, x) -> f(y, x, p), res, u)
end

function _scalar_nlsolve_cache_ad(nest_cache, u, p_nest)
    _p_nest = ForwardDiff.value(p_nest)
    reinit!(nest_cache, ForwardDiff.value.(u), p = _p_nest)
    sol = solve!(nest_cache)
    uu = sol.u
    res = zero(uu)
    f_p = _scalar_nlsolve_∂f_∂p(nest_cache.f, res, uu, _p_nest)
    f_x = _scalar_nlsolve_∂f_∂u(nest_cache.f, res, uu, _p_nest)

    z_arr = -inv(f_x) * f_p

    pp = p_nest
    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if uu isa Number
        partials = sum(sumfun, zip(z_arr, pp))
    elseif _p_nest isa Number
        partials = sumfun((z_arr, pp))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), pp))
    end

    return sol, partials
end

#= function solve_cache!(nest_cache, u::AbstractArray,
                      p_nest::AbstractArray{<:Dual{T, V, P}}) where {T, V, P}

    sol, partials = _scalar_nlsolve_cache_ad(nest_cache, u, p_nest)
    if isdefined(Main, :Infiltrator)
        Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
          end
    #dual_soln = NonlinearSolve.scalar_nlsolve_dual_soln(sol.u, partials, p_nest)
    dual_soln =  map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(sol.u, partials))
    return SciMLBase.build_solution(nest_cache.prob, nest_cache.alg, dual_soln, sol.resid;
                                    sol.retcode)
end =#

function solve_cache!(nest_cache,
                      p_nest::AbstractArray{<:Dual{T, V, P}}) where {T, V, P}
    K = fill(one(eltype(nest_cache.u)), size(nest_cache.u))
    sol, partials = _scalar_nlsolve_cache_ad(nest_cache, K, p_nest)
    #dual_soln = NonlinearSolve.scalar_nlsolve_dual_soln(sol.u, partials, p_nest)
    dual_soln = map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(sol.u, partials))
    return SciMLBase.build_solution(nest_cache.prob, nest_cache.alg, dual_soln, sol.resid;
                                    sol.retcode)
end
