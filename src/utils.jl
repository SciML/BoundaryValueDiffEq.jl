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
