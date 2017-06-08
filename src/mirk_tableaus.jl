immutable MIRKTableau{T<:Number}
    c::Vector{T}
    v::Vector{T}
    b::Vector{T}
    x::Matrix{T}
    K::Matrix{T} # Cache
end

function constructMIRK_IV(T, N)
    c = [0, 1, 1//2, 3//4]
    v = [0, 1, 1//2, 27//32]
    b = [1//6, 1//6, 2//3, 0]
    x = [0      0       0 0
         0      0       0 0
         1//8   -1//8   0 0
         3//64  -9//64  0 0]'
    K = Matrix{T}(N,4)
    MIRKTableau(T.(c),T.(v),T.(b),T.(x),K)
end

constructMIRK{T}(::Type{Val{4}}, N::Integer, A::StridedVecOrMat{T}) = constructMIRK_IV(T, N)
