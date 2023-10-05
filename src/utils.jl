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

# Helpers for IIP/OOP functions
function __sparse_jacobian_cache(::Val{iip}, ad, sd, fn, fx, y) where {iip}
    if iip
        sparse_jacobian_cache(ad, sd, fn, fx, y)
    else
        sparse_jacobian_cache(ad, sd, fn, y; fx)
    end
end
