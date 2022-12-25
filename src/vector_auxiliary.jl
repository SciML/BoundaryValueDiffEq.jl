# Auxiliary functions for working with vector of vectors
function vector_alloc(T, M, N)
    v = Vector{Vector{T}}(undef, N)
    for i in eachindex(v)
        v[i] = zeros(T, M)
    end
    v
end

function vector_alloc(u0::AbstractArray{T}, x) where {T <: AbstractArray}
    @assert length(u0) == length(x)
    deepcopy(u0)
end

function vector_alloc(u0, x)
    [copy(u0) for i in eachindex(x)]
end

function flatten_vector!(dest::T,
                         src::Vector{T2}) where {T <: AbstractArray, T2 <: AbstractArray}
    N = length(src)
    M = length(src[1])
    for i in eachindex(src)
        dest[(((i - 1) * M) + 1):(i * M)] = src[i]
    end
end

function nest_vector!(dest::Vector{T},
                      src::T2) where {T <: AbstractArray, T2 <: AbstractArray}
    M = length(dest[1])
    for i in eachindex(dest)
        copyto!(dest[i], src[((M * (i - 1)) + 1):(M * i)])
    end
end
