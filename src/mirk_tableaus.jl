function constructMIRK_IV(S::BVPSystem{T, U}) where {T, U}
    c = [0, 1, 1 // 2, 3 // 4]
    v = [0, 1, 1 // 2, 27 // 32]
    b = [1 // 6, 1 // 6, 2 // 3, 0]
    x = [0 0 0 0
        0 0 0 0
        1//8 -1//8 0 0
        3//64 -9//64 0 0]
    s = 3
    s_star = 4
    tau = 0.226
    MIRKTableau(T.(c), T.(v), T.(b), T.(x), T(s), T(s_star), T(tau))

end

MIRK_dispatcher(S::BVPSystem, ::Type{Val{4}}) = constructMIRK_IV(S)

function constructMIRK_VI(S::BVPSystem{T, U}) where {T, U}
    c = [0, 1, 1 // 4, 3 // 4, 1 // 2, 7 // 16, 1 // 8, 9 // 16, 3 // 8]
    v = [0, 1, 5 // 32, 27 // 32, 1 // 2, 7 // 16, 1 // 8, 9 // 16, 3 // 8]
    b = [7 // 90, 7 // 90, 16 // 45, 16 // 45, 2 // 15, 0, 0, 0, 0]
    x = [0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        9//64 -3//64 0 0 0 0 0 0 0
        3//64 -9//64 0 0 0 0 0 0 0
        -5//24 5//24 2//3 -2//3 0 0 0 0 0
        1547//32768 -1225//32768 749//4096 -287//2048 -861//16384 0 0 0 0
        83//1536 -13//384 283//1536 -167//1536 -49//512 0 0 0 0
        1225//32768 -1547//32768 287//2048 -749//4096 861//16384 0 0 0 0
        233//3456 -19//1152 0 0 0 -5//72 7//72 -17//216 0]
    s = 5
    s_star = 9
    tau = 0.7156
    MIRKTableau(T.(c), T.(v), T.(b), T.(x), T(s), T(s_star), T(tau))
end

constructMIRK(S::BVPSystem) = MIRK_dispatcher(S, Val{S.order})
MIRK_dispatcher(S::BVPSystem, ::Type{Val{6}}) = constructMIRK_VI(S)
