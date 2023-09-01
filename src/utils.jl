recursive_length(x::Vector{<:AbstractArray}) = sum(length, x)
recursive_length(x::Vector{<:DiffCache}) = sum(length, x)

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

@views function recursive_flatten!(y::AbstractVector, x::Vector{<:DiffCache};
    skip_first::Bool = false)
    i = 0
    x_ = skip_first ? x[2:end] : x
    for xᵢ in x_
        xᵢ_ = get_tmp(xᵢ, y)
        copyto!(y[(i + 1):(i + length(xᵢ_))], xᵢ_)
        i += length(xᵢ_)
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

@views function recursive_unflatten!(y::Vector{<:DiffCache}, x::AbstractVector)
    i = 0
    for yᵢ in y
        yᵢ_ = get_tmp(yᵢ, x)
        copyto!(yᵢ_, x[(i + 1):(i + length(yᵢ_))])
        i += length(yᵢ_)
    end
    return y
end
