recursive_length(x::Vector{<:AbstractArray}) = sum(length, x)
recursive_length(x::Vector{<:MaybeDiffCache}) = sum(xᵢ -> length(xᵢ.u), x)

function recursive_flatten(x::Vector{<:AbstractArray})
    y = similar(first(x), recursive_length(x))
    recursive_flatten!(y, x)
    return y
end

@views function recursive_flatten!(y::AbstractVector, x::Vector{<:AbstractArray})
    i = 0
    for xᵢ in x
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    return y
end
@views function recursive_flatten_twopoint!(y::AbstractVector, x::Vector{<:AbstractArray},
        sizes)
    x_, xiter = Iterators.peel(x)
    copyto!(y[1:prod(sizes[1])], x_[1:prod(sizes[1])])
    i = prod(sizes[1])
    for xᵢ in xiter
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    copyto!(y[(i + 1):(i + prod(sizes[2]))], x_[(end - prod(sizes[2]) + 1):end])
    return y
end

@views function recursive_unflatten!(y::Vector{<:AbstractArray}, x::AbstractVector)
    i = 0
    for yᵢ in y
        copyto!(yᵢ, x[(i + 1):(i + length(yᵢ))])
        i += length(yᵢ)
    end
    return y
end

@views function recursive_unflatten!(y::Vector{<:MaybeDiffCache}, x::AbstractVector)
    return recursive_unflatten!(get_tmp.(y, (x,)), x)
end

function recursive_fill!(y::Vector{<:AbstractArray}, x)
    for yᵢ in y
        fill!(yᵢ, x)
    end
    return y
end

function diff!(dx, x)
    for i in eachindex(dx)
        dx[i] = x[i + 1] - x[i]
    end
    return dx
end

function __maybe_matmul!(z::Array, A, b, α = eltype(z)(1), β = eltype(z)(0))
    mul!(z, A, b, α, β)
end

# NOTE: We can implement it as mul! as above but then we pay the cost of moving
#       `w` to the GPU too many times. Instead if we iterate of w and w′ we save
#       that cost. Our main cost is anyways going to be due to a large `u0` and
#       we are going to use GPUs for that
function __maybe_matmul!(z, A, b, α = eltype(z)(1), β = eltype(z)(0))
    for j in eachindex(b)
        z .= α .* A[:, j] .* b[j] .+ β .* z
    end
    return z
end

## Easier to dispatch
eval_bc_residual(pt, bc::BC, sol, p) where {BC} = eval_bc_residual(pt, bc, sol, p, sol.t)
eval_bc_residual(_, bc::BC, sol, p, t) where {BC} = bc(sol, p, t)
function eval_bc_residual(::TwoPointBVProblem, (bca, bcb)::BC, sol, p, t) where {BC}
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))
    ub = sol isa AbstractVector ? sol[end] : sol(last(t))
    resida = bca(ua, p)
    residb = bcb(ub, p)
    return (resida, residb)
end

function eval_bc_residual!(resid, pt, bc!::BC, sol, p) where {BC}
    return eval_bc_residual!(resid, pt, bc!, sol, p, sol.t)
end
eval_bc_residual!(resid, _, bc!::BC, sol, p, t) where {BC} = bc!(resid, sol, p, t)
@views function eval_bc_residual!(resid, ::TwoPointBVProblem, (bca!, bcb!)::BC, sol, p,
        t) where {BC}
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))
    ub = sol isa AbstractVector ? sol[end] : sol(last(t))
    bca!(resid.resida, ua, p)
    bcb!(resid.residb, ub, p)
    return resid
end
@views function eval_bc_residual!(resid::Tuple, ::TwoPointBVProblem, (bca!, bcb!)::BC, sol,
        p, t) where {BC}
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))
    ub = sol isa AbstractVector ? sol[end] : sol(last(t))
    bca!(resid[1], ua, p)
    bcb!(resid[2], ub, p)
    return resid
end

__append_similar!(::Nothing, n, _) = nothing

# NOTE: We use `last` since the `first` might not conform to the same structure. For eg,
#       in the case of residuals
function __append_similar!(x::AbstractVector{<:AbstractArray}, n, _)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [similar(last(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:MaybeDiffCache}, n, M)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    chunksize = pickchunksize(M * (N + length(x)))
    append!(x, [__maybe_allocate_diffcache(last(x), chunksize) for _ in 1:N])
    return x
end

## Problem with Initial Guess
function __extract_problem_details(prob; kwargs...)
    return __extract_problem_details(prob, prob.u0; kwargs...)
end
### TODO: Support DiffEqArray for non-uniform mesh for multiple shooting
function __extract_problem_details(prob, u0::VectorOfArray; kwargs...)
    # Problem has Initial Guess
    _u0 = first(u0)
    return Val(true), eltype(_u0), length(_u0), (length(u0) - 1), _u0
end
function __extract_problem_details(prob, u0::AbstractArray; dt = 0.0,
        check_positive_dt::Bool = false)
    # Problem does not have Initial Guess
    check_positive_dt && dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    t₀, t₁ = prob.tspan
    return Val(false), eltype(u0), length(u0), Int(cld(t₁ - t₀, dt)), prob.u0
end
function __extract_problem_details(prob, f::F; dt = 0.0,
        check_positive_dt::Bool = false) where {F <: Function}
    # Problem passes in a initial guess function
    check_positive_dt && dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    u0 = __initial_guess(f, prob.p, prob.tspan[1])
    t₀, t₁ = prob.tspan
    return Val(true), eltype(u0), length(u0), Int(cld(t₁ - t₀, dt)), u0
end

function __initial_guess(f::F, p::P, t::T) where {F, P, T}
    if static_hasmethod(f, Tuple{P, T})
        return f(p, t)
    elseif static_hasmethod(f, Tuple{T})
        Base.depwarn("initial guess function must take 2 inputs `(p, t)` instead of just \
                     `t`. The single argument version has been deprecated and will be \
                     removed in the next major release of SciMLBase.", :__initial_guess)
        return f(t)
    else
        throw(ArgumentError("`initial_guess` must be a function of the form `f(p, t)`"))
    end
end

function __initial_state_from_prob(prob::BVProblem, mesh)
    return __initial_state_from_prob(prob, prob.u0, mesh)
end
function __initial_state_from_prob(::BVProblem, u0::AbstractArray, mesh)
    return [copy(vec(u0)) for _ in mesh]
end
function __initial_state_from_prob(::BVProblem, u0::VectorOfArray, _)
    return [copy(vec(u)) for u in u0]
end
function __initial_state_from_prob(prob::BVProblem, f::F, mesh) where {F}
    return [__initial_guess(f, prob.p, t) for t in mesh]
end

function __get_bcresid_prototype(prob::BVProblem, u)
    return __get_bcresid_prototype(prob.problem_type, prob, u)
end
function __get_bcresid_prototype(::TwoPointBVProblem, prob::BVProblem, u)
    prototype = if prob.f.bcresid_prototype !== nothing
        prob.f.bcresid_prototype.x
    else
        first(prob.f.bc)(u, prob.p), last(prob.f.bc)(u, prob.p)
    end
    return prototype, size.(prototype)
end
function __get_bcresid_prototype(::StandardBVProblem, prob::BVProblem, u)
    prototype = prob.f.bcresid_prototype !== nothing ? prob.f.bcresid_prototype :
                __zeros_like(u)
    return prototype, size(prototype)
end

@inline function __fill_like(v, x, args...)
    y = similar(x, args...)
    fill!(y, v)
    return y
end
@inline __zeros_like(args...) = __fill_like(0, args...)
@inline __ones_like(args...) = __fill_like(1, args...)

@inline __safe_vec(x) = vec(x)
@inline __safe_vec(x::Tuple) = mapreduce(__safe_vec, vcat, x)

@inline __vec(x::AbstractArray) = vec(x)
@inline __vec(x::Tuple) = mapreduce(__vec, vcat, x)

# Restructure Non-Vector Inputs
function __vec_f!(du, u, p, t, f!, u_size)
    f!(reshape(du, u_size), reshape(u, u_size), p, t)
    return nothing
end

__vec_f(u, p, t, f, u_size) = vec(f(reshape(u, u_size), p, t))

function __vec_bc!(resid, sol, p, t, bc!, resid_size, u_size)
    bc!(reshape(resid, resid_size), __restructure_sol(sol, u_size), p, t)
    return nothing
end

function __vec_bc!(resid, sol, p, bc!, resid_size, u_size)
    bc!(reshape(resid, resid_size), reshape(sol, u_size), p)
    return nothing
end

__vec_bc(sol, p, t, bc, u_size) = vec(bc(__restructure_sol(sol, u_size), p, t))
__vec_bc(sol, p, bc, u_size) = vec(bc(reshape(sol, u_size), p))

__get_non_sparse_ad(ad::AbstractADType) = ad
function __get_non_sparse_ad(ad::AbstractSparseADType)
    if ad isa AutoSparseForwardDiff
        return AutoForwardDiff{__get_chunksize(ad), typeof(ad.tag)}(ad.tag)
    elseif ad isa AutoSparseEnzyme
        return AutoEnzyme()
    elseif ad isa AutoSparseFiniteDiff
        return AutoFiniteDiff()
    elseif ad isa AutoSparseReverseDiff
        return AutoReverseDiff(ad.compile)
    elseif ad isa AutoSparseZygote
        return AutoZygote()
    else
        throw(ArgumentError("Unknown AD Type"))
    end
end

__get_chunksize(::AutoSparseForwardDiff{CK}) where {CK} = CK

# Restructure Solution
function __restructure_sol(sol::Vector{<:AbstractArray}, u_size)
    return map(Base.Fix2(reshape, u_size), sol)
end

# TODO: Add dispatch for a ODESolution Type as well

# Override the checks for NonlinearFunction
struct __unsafe_nonlinearfunction{iip} end

@inline function __unsafe_nonlinearfunction{iip}(f::F; jac::J = nothing,
        jac_prototype::JP = nothing, colorvec::CV = nothing,
        resid_prototype::RP = nothing) where {iip, F, J, JP, CV, RP}
    return NonlinearFunction{iip, SciMLBase.FullSpecialize, F, Nothing, Nothing, Nothing,
        J, Nothing, Nothing, JP, Nothing, Nothing, Nothing, Nothing, Nothing, CV, Nothing,
        RP}(f, nothing, nothing, nothing, jac, nothing, nothing, jac_prototype, nothing,
        nothing, nothing, nothing, nothing, colorvec, nothing, resid_prototype)
end

@inline __nameof(::T) where {T} = nameof(T)
@inline __nameof(::Type{T}) where {T} = nameof(T)

# Construct the internal NonlinearProblem
@inline function __internal_nlsolve_problem(::BVProblem{uType, tType, iip, nlls},
        resid_prototype, u0, args...; kwargs...) where {uType, tType, iip, nlls}
    if nlls
        return NonlinearLeastSquaresProblem(args...; kwargs...)
    else
        return NonlinearProblem(args...; kwargs...)
    end
end

@inline function __internal_nlsolve_problem(bvp::BVProblem{uType, tType, iip, Nothing},
        resid_prototype, u0, args...; kwargs...) where {uType, tType, iip}
    return __internal_nlsolve_problem(bvp, length(resid_prototype), length(u0), args...;
        kwargs...)
end

@inline function __internal_nlsolve_problem(::BVProblem{uType, tType, iip, Nothing},
        l1::Int, l2::Int, args...; kwargs...) where {uType, tType, iip}
    if l1 != l2
        return NonlinearLeastSquaresProblem(args...; kwargs...)
    else
        return NonlinearProblem(args...; kwargs...)
    end
end

# Populate the `original` and `resid` of the ODESolution
@inline function __update_odesolution(sol::ODESolution{T, N, uType, uType2, DType, tType,
            rateType, P, A, IType, S, AC, R, O}; original = nothing, retcode = sol.retcode,
        resid = nothing) where {T, N, uType, uType2, DType, tType, rateType, P, A, IType,
        S, AC, R, O}
    return ODESolution{T, N, uType, uType2, DType, tType, rateType, P, A, IType, S, AC,
        typeof(resid), typeof(original)}(sol.u, sol.u_analytic, sol.errors, sol.t, sol.k,
        sol.prob, sol.alg, sol.interp, sol.dense, sol.tslocation, sol.stats, sol.alg_choice,
        retcode, resid, original)
end
