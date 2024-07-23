@concrete mutable struct MIRKNCache{iip, T}
    order::Int                 # The order of MIRKN method
    stage::Int                 # The state of MIRKN method
    M::Int                     # The number of equations
    in_size
    f
    bc
    prob                       # BVProblem
    problem_type               # StandardBVProblem
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

function SciMLBase.__init(prob::SecondOrderBVProblem, alg::AbstractMIRKN; dt = 0.0, kwargs...)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    iip = isinplace(prob)
    t₀, t₁ = prob.tspan
    ig, T, M, Nig, X = __extract_problem_details(prob; dt, check_positive_dt = true)
    mesh = __extract_mesh(prob.u0, t₀, t₁, Nig)
    mesh_dt = diff(mesh)
    
    TU = constructMIRKN(alg, T)
   
    # Don't flatten this here, since we need to expand it later if needed
    y₀ = __initial_guess_on_mesh(prob, prob.u0, Nig, prob.p, false)
    chunksize = pickchunksize(M * (2*Nig - 2))
    __alloc = @closure x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

    y = __alloc.(copy.(y₀))
    fᵢ_cache = __alloc(similar(X))
    fᵢ₂_cache = __alloc(similar(X))
    stage = alg_stage(alg)
    bcresid_prototype = __zeros_like(vcat(X, X))
    k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg) for _ in 1:Nig]

    residual = if iip
        __alloc.(copy.(@view(y₀[1:end])))
    else
        nothing
    end

    resid_size = size(bcresid_prototype)
    f, bc = if X isa AbstractVector
        prob.f, prob.f.bc
    elseif iip
        vecf! = @closure (ddu, du, u, p, t) -> __vec_f!(ddu, du, u, p, t, prob.f, size(X))
        vecbc! = if !(prob.problem_type isa TwoPointSecondOrderBVProblem)
            @closure (r, du, u, p, t) -> __vec_so_bc!(r, du, u, p, t, prob.f.bc, resid_size, size(X))
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
        prob.p, alg, TU, bcresid_prototype, mesh, mesh_dt, k_discrete,
        y, y₀, residual, fᵢ_cache, fᵢ₂_cache, resid_size, kwargs)
end

function SciMLBase.solve!(cache::MIRKNCache{iip, T}) where {iip, T}
    (; mesh, M, p, prob, kwargs) = cache
    nlprob = __construct_nlproblem(cache, recursive_flatten(cache.y₀))
    nlsolve_alg = __concrete_nonlinearsolve_algorithm(nlprob, cache.alg.nlsolve)
    sol_nlprob = __solve(
        nlprob, nlsolve_alg; kwargs..., alias_u0 = true)
    recursive_unflatten!(cache.y₀, sol_nlprob.u)
    solu = ArrayPartition.(cache.y₀[1:length(mesh)], cache.y₀[length(mesh)+1:end])
    return SciMLBase.build_solution(prob, cache.alg, mesh, solu; retcode = sol_nlprob.retcode)
end

function __construct_nlproblem(cache::MIRKNCache{iip}, y::AbstractVector) where {iip}
    pt = cache.problem_type

    loss_bc = if iip
        @closure (du, u, p) -> __mirkn_loss_bc!(du, u, p, pt, cache.bc, cache.y, cache.mesh)
    else
        @closure (u, p) -> __mirkn_loss_bc(u, p, pt, cache.bc, cache.y, cache.mesh)
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
            du, u, p, cache.y, pt, cache.bc, cache.residual, cache.mesh, cache)
    else
        @closure (u, p) -> __mirkn_loss(u, p, cache.y, pt, cache.bc, cache.mesh, cache)
    end

    return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss, pt)
end

function __construct_nlproblem(cache::MIRKNCache{iip}, y, loss_bc::BC, loss_collocation::C, loss::LF, ::StandardSecondOrderBVProblem) where {BC, C, LF, iip}
    (; nlsolve, jac_alg) = cache.alg
    
    N = length(cache.mesh)
    resid_bc = cache.bcresid_prototype
    resid_collocation = similar(y, cache.M * (2*N - 2))
    
    L = length(resid_bc)
    lossₚ = (iip ? __Fix3 : Base.Fix2)(loss, cache.p)
#=
    loss_bcₚ = (iip ? __Fix3 : Base.Fix2)(loss_bc, cache.p)
    loss_collocationₚ = (iip ? __Fix3 : Base.Fix2)(loss_collocation, cache.p)

    sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() : NoSparsityDetection()
    cache_bc = __sparse_jacobian_cache(Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bcₚ, resid_bc, y)
    
    sd_collocation = if jac_alg.nonbc_diffmode isa AbstractSparseADType
        J_full_band = BandedMatrix(Ones{eltype(y)}(L + cache.M * (2*N - 2), cache.M * 2*N), (L + 1, cache.M + max(2*cache.M - L, 0)))
            __sparsity_detection_alg(__generate_sparse_jacobian_prototype(
                cache, cache.problem_type, y, y, cache.M, N))
    else
        J_full_band = nothing
        NoSparsityDetection()
    end

    cache_collocation = __sparse_jacobian_cache(
        Val(iip), jac_alg.nonbc_diffmode, sd_collocation,
        loss_collocationₚ, resid_collocation, y)

    J_bc = init_jacobian(cache_bc)
    J_c = init_jacobian(cache_collocation)
    if J_full_band === nothing
        jac_prototype = vcat(J_bc, J_c)
    else
        jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, J_bc)
    end

    jac = if iip
        @closure (J, u, p) -> __mirkn_mpoint_jacobian!(
            J, J_c, u, jac_alg.bc_diffmode, jac_alg.nonbc_diffmode, cache_bc,
            cache_collocation, loss_bcₚ, loss_collocationₚ, resid_bc, resid_collocation, L)
    else
        @closure (u, p) -> __mirkn_mpoint_jacobian(
            jac_prototype, J_c, u, jac_alg.bc_diffmode, jac_alg.nonbc_diffmode,
            cache_bc, cache_collocation, loss_bcₚ, loss_collocationₚ, L)
    end
    
=#
    resid_prototype = vcat(resid_bc, resid_collocation)
    jac = if iip
        @closure (J, u, p) -> FiniteDiff.finite_difference_jacobian!(J, lossₚ, u)
    else
        @closure (u, p) -> FiniteDiff.finite_difference_jacobian(lossₚ, u)
    end
    nlf = __unsafe_nonlinearfunction{iip}(loss; jac = jac, resid_prototype = resid_prototype)

    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __construct_nlproblem(cache::MIRKNCache{iip}, y, loss_bc::BC, loss_collocation::C, loss::LF, ::TwoPointSecondOrderBVProblem) where {BC, C, LF, iip}
    (; nlsolve, jac_alg) = cache.alg
    
    
    N = length(cache.mesh)
    lossₚ = iip ? ((du, u) -> loss(du, u, cache.p)) : (u -> loss(u, cache.p))
    resid = vcat(@view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
    @view(cache.bcresid_prototype[prod(cache.resid_size[1]+1):end]),
    similar(y, cache.M * (2*N - 2)))
#=

    sd = if jac_alg.diffmode isa AbstractSparseADType
        __sparsity_detection_alg(__generate_sparse_jacobian_prototype(
            cache, cache.problem_type,
            @view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
            @view(cache.bcresid_prototype[prod(cache.resid_size[1]+1):end]),
            cache.M, N))
    else
        NoSparsityDetection()
    end
    diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, lossₚ, resid, y)
    jac_prototype = init_jacobian(diffcache)

    jac = if iip
        @closure (J, u, p) -> __mirkn_2point_jacobian!(
            J, u, jac_alg.diffmode, diffcache, lossₚ, resid)
    else
        @closure (u, p) -> __mirkn_2point_jacobian(
            u, jac_prototype, jac_alg.diffmode, diffcache, lossₚ)
    end
=#
    jac = if iip
        @closure (J, u, p) -> FiniteDiff.finite_difference_jacobian!(J, lossₚ, u)
    else
        @closure (u, p) -> FiniteDiff.finite_difference_jacobian(lossₚ, u)
    end
    resid_prototype = copy(resid)
    nlf = __unsafe_nonlinearfunction{iip}(loss; resid_prototype = resid_prototype, jac = jac)#, jac_prototype = jac_prototype)

    return __internal_nlsolve_problem(cache.prob, resid_prototype, y, nlf, y, cache.p)
end

function __mirkn_2point_jacobian!(J, x, diffmode, diffcache, loss_fn::L, resid) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, resid, x)
    return J
end

function __mirkn_2point_jacobian(x, J, diffmode, diffcache, loss_fn::L) where {L}
    sparse_jacobian!(J, diffmode, diffcache, loss_fn, x)
    return J
end

@inline function __internal_nlsolve_problem(
    ::SecondOrderBVProblem{uType, tType, iip, nlls}, resid_prototype,
    u0, args...; kwargs...) where {uType, tType, iip, nlls}
    return NonlinearProblem(args...; kwargs...)
end

function __generate_sparse_jacobian_prototype(cache::MIRKNCache, ya, yb, M, N)
    return __generate_sparse_jacobian_prototype(cache, cache.problem_type, ya, yb, M, N)
end

function __generate_sparse_jacobian_prototype(
        ::MIRKNCache, ::StandardSecondOrderBVProblem, ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (2*N - 2), M * 2 * N), (1, 2M - 1))
    return ColoredMatrix(J_c, matrix_colors(J_c'), matrix_colors(J_c))
end

function __generate_sparse_jacobian_prototype(
        ::MIRKNCache, ::TwoPointSecondOrderBVProblem, ya, yb, M, N)
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (2*N - 2)
    J₂ = M * 2*N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # for underdetermined systems we don't have banded qr implemented. use sparse
    J₁ < J₂ && return ColoredMatrix(sparse(J), matrix_colors(J'), matrix_colors(J))
    return ColoredMatrix(J, matrix_colors(J'), matrix_colors(J))
end

function __mirkn_mpoint_jacobian!(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache, loss_bc::BC,
        loss_collocation::C, resid_bc, resid_collocation, L::Int) where {BC, C}
    sparse_jacobian!(@view(J[1:L, :]), bc_diffmode, bc_diffcache, loss_bc, resid_bc, x)
    sparse_jacobian!(@view(J[(L + 1):end, :]), nonbc_diffmode,
        nonbc_diffcache, loss_collocation, resid_collocation, x)
    return nothing
end

function __mirkn_mpoint_jacobian!(J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode,
        bc_diffcache, nonbc_diffcache, loss_bc::BC, loss_collocation::C,
        resid_bc, resid_collocation, L::Int) where {BC, C}
    J_bc = fillpart(J)
    sparse_jacobian!(J_bc, bc_diffmode, bc_diffcache, loss_bc, resid_bc, x)
    sparse_jacobian!(
        J_c, nonbc_diffmode, nonbc_diffcache, loss_collocation, resid_collocation, x)
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return nothing
end

function __mirkn_mpoint_jacobian(
        J, _, x, bc_diffmode, nonbc_diffmode, bc_diffcache, nonbc_diffcache,
        loss_bc::BC, loss_collocation::C, L::Int) where {BC, C}
    sparse_jacobian!(@view(J[1:L, :]), bc_diffmode, bc_diffcache, loss_bc, x)
    sparse_jacobian!(
        @view(J[(L + 1):end, :]), nonbc_diffmode, nonbc_diffcache, loss_collocation, x)
    return J
end

function __mirkn_mpoint_jacobian(
        J::AlmostBandedMatrix, J_c, x, bc_diffmode, nonbc_diffmode, bc_diffcache,
        nonbc_diffcache, loss_bc::BC, loss_collocation::C, L::Int) where {BC, C}
    J_bc = fillpart(J)
    sparse_jacobian!(J_bc, bc_diffmode, bc_diffcache, loss_bc, x)
    sparse_jacobian!(J_c, nonbc_diffmode, nonbc_diffcache, loss_collocation, x)
    exclusive_bandpart(J) .= J_c
    finish_part_setindex!(J)
    return J
end

@views function __mirkn_loss!(resid, u, p, y, pt::StandardSecondOrderBVProblem, bc::BC, residual, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    eval_bc_residual!(resids[1:2], pt, bc, y_, p, mesh)
    Φ!(resids[3:end], cache, y_, u, p)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(u, p, y, pt::StandardSecondOrderBVProblem, bc::BC, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_bc = eval_bc_residual(pt, bc, y_, p, mesh)
    resid_co = Φ(cache, y_, u, p)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __mirkn_loss!(resid, u, p, y, pt::TwoPointSecondOrderBVProblem, bc!::BC, residual, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resids = [get_tmp(r, u) for r in residual]
    Φ!(resids[3:end], cache, y_, u, p)
    eval_bc_residual!(resids, pt, bc!, y_, p, mesh)
    recursive_flatten!(resid, resids)
    return nothing
end

@views function __mirkn_loss(u, p, y, pt::TwoPointSecondOrderBVProblem, bc!::BC, mesh, cache::MIRKNCache) where {BC}
    y_ = recursive_unflatten!(y, u)
    resid_co = Φ(cache, y_, u, p)
    resid_bc = eval_bc_residual(pt, bc!, y_, p, mesh)
    return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
end

@views function __mirkn_loss_bc!(resid, u, p, pt, bc!::BC, y, mesh) where {BC}
    y_ = recursive_unflatten!(y, u)
    general_eval_bc_residual!(resid, pt, bc!, y_, p, mesh)
    return nothing
end

@views function __mirkn_loss_bc(u, p, pt, bc!::BC, y, mesh) where {BC}
    y_ = recursive_unflatten!(y, u)
    return general_eval_bc_residual(pt, bc!, y_, p, mesh)
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
