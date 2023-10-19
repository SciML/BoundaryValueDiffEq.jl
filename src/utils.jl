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
@views function recursive_flatten_twopoint!(y::AbstractVector, x::Vector{<:AbstractArray})
    x_, xiter = Iterators.peel(x)
    # x_ will be an ArrayPartition
    copyto!(y[1:length(x_.x[1])], x_.x[1])
    i = length(x_.x[1])
    for xᵢ in xiter
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    copyto!(y[(i + 1):(i + length(x_.x[2]))], x_.x[2])
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
eval_bc_residual(pt, bc, sol, p) = eval_bc_residual(pt, bc, sol, p, sol.t)
eval_bc_residual(_, bc, sol, p, t) = bc(sol, p, t)
function eval_bc_residual(::TwoPointBVProblem, (bca, bcb), sol, p, t)
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))
    ub = sol isa AbstractVector ? sol[end] : sol(last(t))
    resid₀ = bca(ua, p)
    resid₁ = bcb(ub, p)
    return ArrayPartition(resid₀, resid₁)
end

eval_bc_residual!(resid, pt, bc!, sol, p) = eval_bc_residual!(resid, pt, bc!, sol, p, sol.t)
eval_bc_residual!(resid, _, bc!, sol, p, t) = bc!(resid, sol, p, t)
@views function eval_bc_residual!(resid, ::TwoPointBVProblem, (bca!, bcb!), sol, p, t)
    ua = sol isa AbstractVector ? sol[1] : sol(first(t))
    ub = sol isa AbstractVector ? sol[end] : sol(last(t))
    bca!(resid.x[1], ua, p)
    bcb!(resid.x[2], ub, p)
    return resid
end

__append_similar!(::Nothing, n, _) = nothing

function __append_similar!(x::AbstractVector{<:AbstractArray}, n, _, TU)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [similar(first(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:MaybeDiffCache}, n, M, TU)
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    chunksize = pickchunksize(M * (N + length(x)))
    append!(x, [__maybe_allocate_diffcache(first(x), chunksize) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:AbstractArray}, n, _) 
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [similar(first(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:MaybeDiffCache}, n, M) 
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    chunksize = pickchunksize(M * (N + length(x)))
    append!(x, [__maybe_allocate_diffcache(first(x), chunksize) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:AbstractArray}, n, _, TU::RKTableau{false})
    @unpack s = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [similar(first(x)) for _ in 1:N])
    return x
end

function __append_similar!(x::AbstractVector{<:MaybeDiffCache}, n, M, TU::RKTableau{false})
    @unpack s = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    chunksize = isa(TU, RKTableau{false}) ? pickchunksize(M * (N + length(x) * (s + 1))) : pickchunksize(M * (N + length(x)))
    append!(x, [__maybe_allocate_diffcache(first(x), chunksize) for _ in 1:N])
    return x
end

## Problem with Initial Guess
function __extract_problem_details(prob; kwargs...)
    return __extract_problem_details(prob, prob.u0; kwargs...)
end
function __extract_problem_details(prob, u0::AbstractVector{<:AbstractArray}; kwargs...)
    # Problem has Initial Guess
    _u0 = first(u0)
    return True(), eltype(_u0), length(_u0), (length(u0) - 1), _u0
end
function __extract_problem_details(prob, u0; dt = 0.0, check_positive_dt::Bool = false)
    # Problem does not have Initial Guess
    check_positive_dt && dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    t₀, t₁ = prob.tspan
    return False(), eltype(u0), length(u0), Int(cld(t₁ - t₀, dt)), prob.u0
end

__initial_state_from_prob(prob::BVProblem, mesh) = __initial_state_from_prob(prob.u0, mesh)
__initial_state_from_prob(u0::AbstractArray, mesh) = [copy(vec(u0)) for _ in mesh]
function __initial_state_from_prob(u0::AbstractVector{<:AbstractVector}, _)
    return [copy(vec(u)) for u in u0]
end

function __get_bcresid_prototype(prob::BVProblem, u)
    return __get_bcresid_prototype(prob.problem_type, prob, u)
end
function __get_bcresid_prototype(::TwoPointBVProblem, prob::BVProblem, u)
    prototype = if isinplace(prob)
        prob.f.bcresid_prototype
    elseif prob.f.bcresid_prototype !== nothing
        prob.f.bcresid_prototype
    else
        ArrayPartition(first(prob.f.bc)(u, prob.p), last(prob.f.bc)(u, prob.p))
    end
    return prototype, size.(prototype.x)
end
function __get_bcresid_prototype(::StandardBVProblem, prob::BVProblem, u)
    prototype = prob.f.bcresid_prototype !== nothing ? prob.f.bcresid_prototype :
                __zeros_like(u)
    return prototype, size(prototype)
end

function __fill_like(v, x, args...)
    y = similar(x, args...)
    fill!(y, v)
    return y
end
__zeros_like(args...) = __fill_like(0, args...)
__ones_like(args...) = __fill_like(1, args...)

__safe_reshape(x, args...) = reshape(x, args...)
function __safe_reshape(x::ArrayPartition, sizes::NTuple)
    return ArrayPartition(__safe_reshape.(x.x, sizes))
end
