# Auxiliary functions for working with vector of vectors
function vector_alloc(T, M, N)
    v = Vector{Vector{T}}(N)
    for i in eachindex(v)
        v[i] = zeros(T, M)
    end
    v
end

flatten_vector{T}(V::Vector{Vector{T}}) = vcat(V...)

function nest_vector{T}(v::Vector{T}, M, N)
    V = vector_alloc(T, M, N)
    for i in eachindex(V)
        copy!(V[i], v[(M*(i-1))+1:(M*i)])
    end
    V
end

