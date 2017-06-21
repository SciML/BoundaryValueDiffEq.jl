function constructMIRK_IV{T}(S::BVPSystem{T})
    c = [0, 1, 1//2, 3//4]
    v = [0, 1, 1//2, 27//32]
    b = [1//6, 1//6, 2//3, 0]
    x = [0      0       0 0
         0      0       0 0
         1//8   -1//8   0 0
         3//64  -9//64  0 0]
    K = vector_alloc(T, S.M, 4)
    MIRKTableau(T.(c),T.(v),T.(b),T.(x),K)
end

constructMIRK{T}(S::BVPSystem{T}) = MIRK_dispatcher(S, Val{S.order})
MIRK_dispatcher{T}(S::BVPSystem{T}, ::Type{Val{4}}) = constructMIRK_IV(S)
