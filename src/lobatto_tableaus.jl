for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIb$(order)")
    f = Symbol("constructLobattoIIIb$(order)")
    @eval constructMIRK(::$(alg), ::Type{T}) where {T} = $(f)(T)
end

function constructLobattoIIIb2(::Type{T}) where {T}
    # RK coefficients tableau
    s = 2
    c = [0, 1]
    v = [0, 0]
    b = [1 // 2, 1 // 2]
    x = [1//2 0
         1//2 0]

    # Interpolant tableau
    #= s_star = 3
    c_star = [1]
    v_star = [1]
    x_star = [0, 0, 0]
    τ_star = 0.25 =#

    TU = ITU = MIRKTableau(Int64(s), T.(c), T.(v), T.(b), T.(x))
    # ITU = MIRKInterpTableau(Int64(s_star), T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructLobattoIIIb3(::Type{T}) where {T}
    # RK coefficients tableau
    s = 3
    c = [0, 1 // 2, 1]
    v = [0, 0, 0, 0]
    b = [1 // 6, 2 // 3, 1 // 6]
    x = [1//6 -1//6 0
         1//6 1//3 0
         1//6 5//6 0]

    # Interpolant tableau
    #= s_star = 4
    c_star = [3 // 4]
    v_star = [27 // 32]
    x_star = [3 // 64, -9 // 64, 0, 0]
    τ_star = 0.226 =#

    TU = ITU = MIRKTableau(s, T.(c), T.(v), T.(b), T.(x))
    # ITU = MIRKInterpTableau(s_star, T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructLobattoIIIb4(::Type{T}) where {T}
    # RK coefficients tableau
    s = 4
    c = [0, 1 // 2 - Rational(√5)//10, 1 // 2 + Rational(√5)//10, 1]
    v = [0, 0, 0, 0]
    b = [1 // 12, 5 // 12, 5 // 12, 1 // 12]
    x = [1 // 12 (-1 - Rational(√5))//24 (-1 + Rational(√5))//24 0
         1 // 12 (25 + Rational(√5))//120 (25 - 13*Rational(√5))//120 0
         1 // 12 (25 + 13*Rational(√5))//120 (25 - Rational(√5))//120 0
         1 // 12 (11 - Rational(√5))//24 (11 + Rational(√5))//24 0]

    # Interpolant tableau
    #= s_star = 6
    c_star = [4 // 5, 13 // 23]
    v_star = [4 // 5, 13 // 23]
    x_star = [14//1125 -74//875 -128//3375 104//945 0 0
        1//2 4508233//1958887 48720832//2518569 -27646420//17629983 -11517095//559682 0]
    τ_star = 0.3 =#

    TU = ITU = MIRKTableau(s, T.(c), T.(v), T.(b), T.(x))
    #ITU = MIRKInterpTableau(s_star, T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end

function constructLobattoIIIb5(::Type{T}) where {T}
    # RK coefficients tableau
    s = 5
    c = [0, 1 // 2 - Rational(√21)//14, 1 // 2, 1 // 2 + Rational(√21)//14, 1]
    v = [0, 0, 0, 0, 0]
    b = [1 // 20, 49 // 180, 16 // 45, 49 // 180, 1 // 20]
    x = [1 // 20 (-7 - Rational(√21))//120 1 // 15 (-7 + Rational(√21))//120 0
         1 // 20 (343 + 9*Rational(√21))//2520 (56 - 15*Rational(√21))//315 (343 - 69*Rational(√21))//2520 0
         1 // 20 (49 + 12*Rational(√21))//360 8//45 (49 - 12*Rational(√21))//360 0
         1 // 20 (343 + 69*Rational(√21))//2520 (56 + 15*Rational(√21))//315 (343 - 9*Rational(√21))//2520 0
         1 // 20 (119 - 3*Rational(√21))//360 13//45 (119 + 3*Rational(√21))//360 0]

    #= # Interpolant tableau
    s_star = 9
    c_star = [7 // 16, 3 // 8, 9 // 16, 1 // 8]
    v_star = [7 // 16, 3 // 8, 9 // 16, 1 // 8]
    x_star = [1547//32768 -1225//32768 749//4096 -287//2048 -861//16384 0 0 0 0
              83//1536 -13//384 283//1536 -167//1536 -49//512 0 0 0 0
              1225//32768 -1547//32768 287//2048 -749//4096 861//16384 0 0 0 0
              233//3456 -19//1152 0 0 0 -5//72 7//72 -17//216 0]
    τ_star = 0.7156 =#

    TU = ITU = MIRKTableau(s, T.(c), T.(v), T.(b), T.(x))
    #ITU = MIRKInterpTableau(s_star, T.(c_star), T.(v_star), T.(x_star), T(τ_star))
    return TU, ITU
end
