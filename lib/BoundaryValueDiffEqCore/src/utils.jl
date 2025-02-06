recursive_length(x::Vector{<:AbstractArray}) = sum(length, x)
recursive_length(x::Vector{<:MaybeDiffCache}) = sum(xᵢ -> length(xᵢ.u), x)

function recursive_flatten(x::Vector{<:AbstractArray})
    y = zero(first(x), recursive_length(x))
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
@views function recursive_flatten_twopoint!(
        y::AbstractVector, x::Vector{<:AbstractArray}, sizes)
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

@views function recursive_unflatten!(y::AbstractVectorOfArray, x::AbstractVector)
    i = 0
    for yᵢ in y
        copyto!(yᵢ, x[(i + 1):(i + length(yᵢ))])
        i += length(yᵢ)
    end
    return y
end

function diff!(dx, x)
    for i in eachindex(dx)
        dx[i] = x[i + 1] - x[i]
    end
    return dx
end

function __maybe_matmul!(z::AbstractArray, A, b, α = eltype(z)(1), β = eltype(z)(0))
    mul!(z, A, b, α, β)
end

# NOTE: We can implement it as mul! as above but then we pay the cost of moving
#       `w` to the GPU too many times. Instead if we iterate of w and w′ we save
#       that cost. Our main cost is anyways going to be due to a large `u0` and
#       we are going to use GPUs for that
@views function __maybe_matmul!(z, A, b, α = eltype(z)(1), β = eltype(z)(0))
    @simd ivdep for j in eachindex(b)
        @inbounds @. z = α * A[:, j] * b[j] + β * z
    end
    return z
end

## Easier to dispatch
eval_bc_residual(pt, bc::BC, sol, p) where {BC} = eval_bc_residual(pt, bc, sol, p, sol.t)
eval_bc_residual(_, bc::BC, sol, p, t) where {BC} = bc(sol, p, t)
function eval_bc_residual(::TwoPointBVProblem, (bca, bcb)::BC, sol, p, t) where {BC}
    ua = sol isa VectorOfArray ? sol[:, 1] : sol(first(t))
    ub = sol isa VectorOfArray ? sol[:, end] : sol(last(t))
    resida = bca(ua, p)
    residb = bcb(ub, p)
    return (resida, residb)
end

function eval_bc_residual!(resid, pt, bc!::BC, sol, p) where {BC}
    return eval_bc_residual!(resid, pt, bc!, sol, p, sol.t)
end
eval_bc_residual!(resid, _, bc!::BC, sol, p, t) where {BC} = bc!(resid, sol, p, t)
@views function eval_bc_residual!(
        resid, ::TwoPointBVProblem, (bca!, bcb!)::BC, sol, p, t) where {BC}
    ua = sol isa VectorOfArray ? sol[:, 1] : sol(first(t))
    ub = sol isa VectorOfArray ? sol[:, end] : sol(last(t))
    bca!(resid.resida, ua, p)
    bcb!(resid.residb, ub, p)
    return resid
end
@views function eval_bc_residual!(
        resid::Tuple, ::TwoPointBVProblem, (bca!, bcb!)::BC, sol, p, t) where {BC}
    ua = sol isa VectorOfArray ? sol[:, 1] : sol(first(t))
    ub = sol isa VectorOfArray ? sol[:, end] : sol(last(t))
    bca!(resid[1], ua, p)
    bcb!(resid[2], ub, p)
    return resid
end

function eval_bc_residual(::StandardSecondOrderBVProblem, bc::BC, y, dy, p, t) where {BC}
    res_bc = bc(dy, y, p, t)
    return res_bc
end
function eval_bc_residual(
        ::TwoPointSecondOrderBVProblem, (bca, bcb)::BC, sol, p, t) where {BC}
    M = length(sol[1])
    L = length(t)
    ua = sol isa VectorOfArray ? sol[:, 1] : sol(first(t))[1:M]
    ub = sol isa VectorOfArray ? sol[:, L] : sol(last(t))[1:M]
    dua = sol isa VectorOfArray ? sol[:, L + 1] : sol(first(t))[(M + 1):end]
    dub = sol isa VectorOfArray ? sol[:, end] : sol(last(t))[(M + 1):end]
    return vcat(bca(dua, ua, p), bcb(dub, ub, p))
end

function eval_bc_residual!(resid, ::StandardBVProblem, bc!::BC, sol, p, t) where {BC}
    bc!(resid, sol, p, t)
end

function eval_bc_residual!(
        resid, ::StandardSecondOrderBVProblem, bc!::BC, sol, dsol, p, t) where {BC}
    M = length(sol[1])
    res_bc = vcat(resid[1], resid[2])
    bc!(res_bc, dsol, sol, p, t)
    copyto!(resid[1], res_bc[1:M])
    copyto!(resid[2], res_bc[(M + 1):end])
end

function eval_bc_residual!(
        resid, ::TwoPointSecondOrderBVProblem, (bca!, bcb!)::BC, sol, p, t) where {BC}
    M = length(sol[1])
    L = length(t)
    ua = sol isa VectorOfArray ? sol[:, 1] : sol(first(t))[1:M]
    ub = sol isa VectorOfArray ? sol[:, L] : sol(last(t))[1:M]
    dua = sol isa VectorOfArray ? sol[:, L + 1] : sol(first(t))[(M + 1):end]
    dub = sol isa VectorOfArray ? sol[:, end] : sol(last(t))[(M + 1):end]
    bca!(resid[1], dua, ua, p)
    bcb!(resid[2], dub, ub, p)
end

__append_similar!(::Nothing, n, _) = nothing

# NOTE: We use `last` since the `first` might not conform to the same structure. For eg,
#       in the case of residuals
function __append_similar!(x::AbstractVector{<:AbstractArray}, n, _)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [zero(last(x)) for _ in 1:N])
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

__append_similar!(::Nothing, n, _, _) = nothing
function __append_similar!(x::AbstractVectorOfArray, n, _)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, VectorOfArray([similar(last(x)) for _ in 1:N]))
    return x
end

## Problem with Initial Guess
function __extract_problem_details(prob; kwargs...)
    return __extract_problem_details(prob, prob.u0; kwargs...)
end
function __extract_problem_details(prob, u0::AbstractVector{<:AbstractArray}; kwargs...)
    # Problem has Initial Guess
    _u0 = first(u0)
    return Val(true), eltype(_u0), length(_u0), (length(u0) - 1), _u0
end
function __extract_problem_details(prob, u0::AbstractVectorOfArray; kwargs...)
    # Problem has Initial Guess
    _u0 = first(u0.u)
    return Val(true), eltype(_u0), length(_u0), (length(u0.u) - 1), _u0
end
function __extract_problem_details(
        prob, u0::AbstractArray; dt = 0.0, check_positive_dt::Bool = false)
    # Problem does not have Initial Guess
    check_positive_dt && dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    t₀, t₁ = prob.tspan
    return Val(false), eltype(u0), length(u0), Int(cld(t₁ - t₀, dt)), prob.u0
end
function __extract_problem_details(
        prob, f::F; dt = 0.0, check_positive_dt::Bool = false) where {F <: Function}
    # Problem passes in a initial guess function
    check_positive_dt && dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    u0 = __initial_guess(f, prob.p, prob.tspan[1])
    t₀, t₁ = prob.tspan
    return Val(true), eltype(u0), length(u0), Int(cld(t₁ - t₀, dt)), u0
end

function __extract_problem_details(
        prob, u0::SciMLBase.ODESolution; dt = 0.0, check_positive_dt::Bool = false)
    # Problem passes in a initial guess function
    check_positive_dt && dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    _u0 = first(u0.u)
    _t = u0.t
    return Val(true), eltype(_u0), length(_u0), (length(_t) - 1), _u0
end

function __initial_guess(f::F, p::P, t::T) where {F, P, T}
    if hasmethod(f, Tuple{P, T})
        return f(p, t)
    elseif hasmethod(f, Tuple{T})
        Base.depwarn("initial guess function must take 2 inputs `(p, t)` instead of just \
                     `t`. The single argument version has been deprecated and will be \
                     removed in the next major release of SciMLBase.",
            :__initial_guess)
        return f(t)
    else
        throw(ArgumentError("`initial_guess` must be a function of the form `f(p, t)`"))
    end
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
    prototype = prob.f.bcresid_prototype !== nothing ? prob.f.bcresid_prototype : zero(u)
    return prototype, size(prototype)
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

@inline function __similar(x, args...)
    y = similar(x, args...)
    return zero(y)
end

@inline function __fill_like(v, x, args...)
    y = __similar(x, args...)
    fill!(y, v)
    return y
end

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
    bc!(reshape(resid, resid_size), sol, p, t)
    return nothing
end

function __vec_bc!(resid, sol, p, bc!, resid_size, u_size)
    bc!(reshape(resid, resid_size), reshape(sol, u_size), p)
    return nothing
end

__vec_bc(sol, p, t, bc, u_size) = vec(bc(sol, p, t))
__vec_bc(sol, p, bc, u_size) = vec(bc(reshape(sol, u_size), p))

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

@inline __get_non_sparse_ad(ad::AbstractADType) = ad
@inline __get_non_sparse_ad(ad::AutoSparse) = ADTypes.dense_ad(ad)

# Restructure Solution
function __restructure_sol(sol::AbstractVectorOfArray, u_size)
    (size(first(sol)) == u_size) && return sol
    return VectorOfArray(map(Base.Fix2(reshape, u_size), sol))
end
function __restructure_sol(sol::AbstractArray{<:AbstractArray}, u_size)
    (size(first(sol)) == u_size) && return sol
    return map(Base.Fix2(reshape, u_size), sol)
end

# Construct the internal NonlinearProblem
@inline function __internal_nlsolve_problem(
        ::BVProblem{uType, tType, iip, nlls}, resid_prototype,
        u0, args...; kwargs...) where {uType, tType, iip, nlls}
    if nlls
        return NonlinearLeastSquaresProblem(args...; kwargs...)
    else
        return NonlinearProblem(args...; kwargs...)
    end
end

@inline function __internal_nlsolve_problem(
        bvp::BVProblem{uType, tType, iip, Nothing}, resid_prototype,
        u0, args...; kwargs...) where {uType, tType, iip}
    return __internal_nlsolve_problem(
        bvp, length(resid_prototype), length(u0), args...; kwargs...)
end

@inline function __internal_nlsolve_problem(
        ::BVProblem{uType, tType, iip, Nothing}, l1::Int,
        l2::Int, args...; kwargs...) where {uType, tType, iip}
    if l1 != l2
        return NonlinearLeastSquaresProblem(args...; kwargs...)
    else
        return NonlinearProblem(args...; kwargs...)
    end
end

@inline function __internal_nlsolve_problem(
        ::SecondOrderBVProblem{uType, tType, iip, nlls}, resid_prototype,
        u0, args...; kwargs...) where {uType, tType, iip, nlls}
    return NonlinearProblem(args...; kwargs...)
end

# Handling Initial Guesses
"""
    __extract_u0(u₀, t₀)

Takes the input initial guess and returns the value at the starting mesh point.
"""
@inline __extract_u0(u₀::AbstractVector{<:AbstractArray}, p, t₀) = u₀[1]
@inline __extract_u0(u₀::VectorOfArray, p, t₀) = u₀[:, 1]
@inline __extract_u0(u₀::DiffEqArray, p, t₀) = u₀.u[1]
@inline __extract_u0(u₀::F, p, t₀) where {F <: Function} = __initial_guess(u₀, p, t₀)
@inline __extract_u0(u₀::AbstractArray, p, t₀) = u₀
@inline __extract_u0(u₀::SciMLBase.ODESolution, p, t₀) = u₀.u[1]
@inline __extract_u0(u₀::T, p, t₀) where {T} = error("`prob.u0::$(T)` is not supported.")

"""
    __extract_mesh(u₀, t₀, t₁, n)

Takes the input initial guess and returns the mesh.
"""
@inline __extract_mesh(u₀, t₀, t₁, n::Int) = collect(range(t₀; stop = t₁, length = n + 1))
@inline __extract_mesh(u₀, t₀, t₁, dt::Number) = collect(t₀:dt:t₁)
@inline __extract_mesh(u₀::DiffEqArray, t₀, t₁, ::Int) = u₀.t
@inline __extract_mesh(u₀::DiffEqArray, t₀, t₁, ::Number) = u₀.t
@inline __extract_mesh(u₀::SciMLBase.ODESolution, t₀, t₁, ::Int) = u₀.t
@inline __extract_mesh(u₀::SciMLBase.ODESolution, t₀, t₁, ::Number) = u₀.t

"""
    __has_initial_guess(u₀) -> Bool

Returns `true` if the input has an initial guess.
"""
@inline __has_initial_guess(u₀::AbstractVector{<:AbstractArray}) = true
@inline __has_initial_guess(u₀::VectorOfArray) = true
@inline __has_initial_guess(u₀::DiffEqArray) = true
@inline __has_initial_guess(u₀::SciMLBase.ODESolution) = true
@inline __has_initial_guess(u₀::F) where {F} = true
@inline __has_initial_guess(u₀::AbstractArray) = false

"""
    __initial_guess_length(u₀) -> Int

Returns the length of the initial guess. If the initial guess is a function or no initial
guess is supplied, it returns `-1`.
"""
@inline __initial_guess_length(u₀::AbstractVector{<:AbstractArray}) = length(u₀)
@inline __initial_guess_length(u₀::VectorOfArray) = length(u₀)
@inline __initial_guess_length(u₀::DiffEqArray) = length(u₀.t)
@inline __initial_guess_length(u₀::SciMLBase.ODESolution) = length(u₀.t)
@inline __initial_guess_length(u₀::F) where {F} = -1
@inline __initial_guess_length(u₀::AbstractArray) = -1

"""
    __flatten_initial_guess(u₀) -> Union{AbstractMatrix, AbstractVector, Nothing}

Flattens the initial guess into a matrix. For a function `u₀`, it returns `nothing`. For no
initial guess, it returns `vec(u₀)`.
"""
@inline __flatten_initial_guess(u₀::AbstractVector{<:AbstractArray}) = mapreduce(
    vec, hcat, u₀)
@inline __flatten_initial_guess(u₀::VectorOfArray) = mapreduce(vec, hcat, u₀.u)
@inline __flatten_initial_guess(u₀::DiffEqArray) = mapreduce(vec, hcat, u₀.u)
@inline __flatten_initial_guess(u₀::SciMLBase.ODESolution) = mapreduce(vec, hcat, u₀.u)
@inline __flatten_initial_guess(u₀::AbstractArray) = vec(u₀)
@inline __flatten_initial_guess(u₀::F) where {F} = nothing

"""
    __initial_guess_on_mesh(u₀, mesh, p, alias_u0::Bool)

Returns the initial guess on the mesh. For `DiffEqArray` assumes that the mesh is the same
as the mesh of the `DiffEqArray`.
"""
@inline function __initial_guess_on_mesh(u₀::AbstractVector{<:AbstractArray}, _, p)
    return VectorOfArray([copy(vec(u)) for u in u₀])
end
@inline function __initial_guess_on_mesh(u₀::VectorOfArray, _, p)
    return copy(u₀)
end
@inline function __initial_guess_on_mesh(u₀::DiffEqArray, mesh, p)
    return copy(u₀)
end
@inline function __initial_guess_on_mesh(u₀::SciMLBase.ODESolution, mesh, p)
    return copy(VectorOfArray(u₀.u))
end
@inline function __initial_guess_on_mesh(u₀::AbstractArray, mesh, p)
    return VectorOfArray([copy(vec(u₀)) for _ in mesh])
end
@inline function __initial_guess_on_mesh(u₀::F, mesh, p) where {F}
    return VectorOfArray([vec(__initial_guess(u₀, p, t)) for t in mesh])
end
@inline function __initial_guess_on_mesh(
        prob::SecondOrderBVProblem, u₀::AbstractArray, Nig, p, alias_u0::Bool)
    return VectorOfArray([copy(vec(u₀)) for _ in 1:(2 * (Nig + 1))])
end

# Construct BVP Solution
function __build_solution(prob::AbstractBVProblem, odesol, nlsol)
    retcode = ifelse(SciMLBase.successful_retcode(nlsol), odesol.retcode, nlsol.retcode)
    return SciMLBase.solution_new_original_retcode(odesol, nlsol, retcode, nlsol.resid)
end

# Fix3
@concrete struct __Fix3
    f
    x
end

@inline (f::__Fix3{F})(a, b) where {F} = f.f(a, b, f.x)

get_dense_ad(::Nothing) = nothing
get_dense_ad(ad) = ad
get_dense_ad(ad::AutoSparse) = ADTypes.dense_ad(ad)
