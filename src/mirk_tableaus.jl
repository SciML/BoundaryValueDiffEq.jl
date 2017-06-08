immutable MIRKTableau{T<:Number}
    c::Vector{T}
    v::Vector{T}
    b::Vector{T}
    x::Matrix{T}
    order::Int
end

function constructMIRK_IV{T}(::T)
    c = [0, 1, 1//2, 3//4]
    v = [0, 1, 1//2, 27//32]
    b = [1//6, 1//6, 2//3, 0]
    x = [0      0       0 0
         0      0       0 0
         1//8   -1//8   0 0
         3//64  -9//64  0 0]
    MIRKTableau(c,v,b,x,3)
end
