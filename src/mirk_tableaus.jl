function constructMIRK_IV(S::BVPSystem{T,U}) where {T,U}
    c = [0, 1, 1//2, 3//4]
    v = [0, 1, 1//2, 27//32]
    b = [1//6, 1//6, 2//3, 0]
    x = [0      0       0 0
         0      0       0 0
         1//8   -1//8   0 0
         3//64  -9//64  0 0]
    MIRKTableau(T.(c),T.(v),T.(b),T.(x))
end

constructMIRK(S::BVPSystem) = MIRK_dispatcher(S, Val{S.order})
MIRK_dispatcher(S::BVPSystem, ::Type{Val{4}}) = constructMIRK_IV(S)
