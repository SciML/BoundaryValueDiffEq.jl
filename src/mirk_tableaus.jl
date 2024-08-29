for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")
    f = Symbol("constructMIRK$(order)")
    @eval constructMIRK(::$(alg), ::Type{T}) where {T} = $(f)(T)
end

function constructMIRK2(::Type{T}) where {T}
    # RK coefficients tableau
    s = 1
    c = [1 // 2]
    v = [1 // 2]
    b = [1]
    x = [0]

    # Interpolant tableau
    s_star = 3
    c_star = [0, 1]
    v_star = [0, 1]
    x_star = [0, 0, 0, 0]
    τ_star = 0.25

    TU = MIRKTableau(Int64(s), T.(c), T.(v), T.(b), T.(x))
    ITU = MIRKInterpTableau(Int64(s_star), T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructMIRK3(::Type{T}) where {T}
    # RK coefficients tableau
    s = 2
    c = [0, 2 // 3]
    v = [0, 4 // 9]
    b = [1 // 4, 3 // 4]
    x = [0 0
         2//9 0]

    # Interpolant tableau
    s_star = 3
    c_star = [1]
    v_star = [1]
    x_star = [0, 0, 0]
    τ_star = 0.25

    TU = MIRKTableau(Int64(s), T.(c), T.(v), T.(b), T.(x))
    ITU = MIRKInterpTableau(Int64(s_star), T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructMIRK4(::Type{T}) where {T}
    # RK coefficients tableau
    s = 3
    c = [0, 1, 1 // 2, 3 // 4]
    v = [0, 1, 1 // 2, 27 // 32]
    b = [1 // 6, 1 // 6, 2 // 3, 0]
    x = [0 0 0 0
         0 0 0 0
         1//8 -1//8 0 0]

    # Interpolant tableau
    s_star = 4
    c_star = [3 // 4]
    v_star = [27 // 32]
    x_star = [3 // 64, -9 // 64, 0, 0]
    τ_star = 0.226

    TU = MIRKTableau(s, T.(c), T.(v), T.(b), T.(x))
    ITU = MIRKInterpTableau(s_star, T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructMIRK5(::Type{T}) where {T}
    # RK coefficients tableau
    s = 4
    c = [0, 1, 3 // 4, 3 // 10]
    v = [0, 1, 27 // 32, 837 // 1250]
    b = [5 // 54, 1 // 14, 32 // 81, 250 // 567]
    x = [0 0 0 0
         0 0 0 0
         3//64 -9//64 0 0
         21//1000 63//5000 -252//625 0]

    # Interpolant tableau
    s_star = 6
    c_star = [4 // 5, 13 // 23]
    v_star = [4 // 5, 13 // 23]
    x_star = [14//1125 -74//875 -128//3375 104//945 0 0
              1//2 4508233//1958887 48720832//2518569 -27646420//17629983 -11517095//559682 0]
    τ_star = 0.3

    TU = MIRKTableau(s, T.(c), T.(v), T.(b), T.(x))
    ITU = MIRKInterpTableau(s_star, T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructMIRK6(::Type{T}) where {T}
    # RK coefficients tableau
    s = 5
    c = [0, 1, 1 // 4, 3 // 4, 1 // 2]
    v = [0, 1, 5 // 32, 27 // 32, 1 // 2]
    b = [7 // 90, 7 // 90, 16 // 45, 16 // 45, 2 // 15, 0, 0, 0, 0]
    x = [0 0 0 0 0
         0 0 0 0 0
         9//64 -3//64 0 0 0
         3//64 -9//64 0 0 0
         -5//24 5//24 2//3 -2//3 0]

    # Interpolant tableau
    s_star = 9
    c_star = [7 // 16, 3 // 8, 9 // 16, 1 // 8]
    v_star = [7 // 16, 3 // 8, 9 // 16, 1 // 8]
    x_star = [1547//32768 -1225//32768 749//4096 -287//2048 -861//16384 0 0 0 0
              83//1536 -13//384 283//1536 -167//1536 -49//512 0 0 0 0
              1225//32768 -1547//32768 287//2048 -749//4096 861//16384 0 0 0 0
              233//3456 -19//1152 0 0 0 -5//72 7//72 -17//216 0]
    τ_star = 0.7156

    TU = MIRKTableau(s, T.(c), T.(v), T.(b), T.(x))
    ITU = MIRKInterpTableau(s_star, T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end
