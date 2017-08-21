# Auxiliary functions for working with vector of vectors
function vector_alloc(T, M, N)
    v = Vector{Vector{T}}(N)
    for i in eachindex(v)
        v[i] = zeros(T, M)
    end
    v
end

function flatten_vector!{T<:AbstractArray}(dest::T, src::Vector{T})
    N = length(src)
    M = length(src[1])
    for i in eachindex(src)
        dest[((i-1)*M)+1:i*M] = src[i]
    end
end

function nest_vector!{T<:AbstractArray}(dest::Vector{T}, src::T)
    M = length(dest[1])
    for i in eachindex(dest)
        copy!(dest[i], src[(M*(i-1))+1:(M*i)])
    end
end
