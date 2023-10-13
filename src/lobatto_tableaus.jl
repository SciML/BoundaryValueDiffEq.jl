# LobattoIIIa
for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIa$(order)")
    f = Symbol("constructLobattoIIIa$(order)")
    @eval constructRK(_alg::$(alg), ::Type{T}) where {T} = $(f)(T, _alg.nested_nlsolve)
end

function constructLobattoIIIa2(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 2
    a = [0 0
         1//2 1//2]
    c = [0, 1]
    b = [1 // 2, 1 // 2]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [3 // 8, 1 // 8]
    poly_max = 0.25
    dn_coeffs = [-1, 1, 1]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

function constructLobattoIIIa3(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 3
    a = [0 0 0
         5//24 1//3 -1//24
         1//6 2//3 1//6]
    c = [0, 1 // 2, 1]
    b = [1 // 6, 2 // 3, 1 // 6]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [0.20833333333333337, 0.33333333333333337, -0.04166666666666667]
    poly_max = 0.048112522432468816
    dn_coeffs = [6, -12, 6, 1.5]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

function constructLobattoIIIa4(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 4
    a = [0 0 0 0
         (11 + Rational(√5))//120 (25 - Rational(√5))//120 (25 - 13 * Rational(√5))//120 (-1 + Rational(√5))//120
         (11 - Rational(√5))//120 (25 + 13 * Rational(√5))//120 (25 + Rational(√5))//120 (-1 - Rational(√5))//120
         1//12 5//12 5//12 1//12]
    c = [0, 1 // 2 - Rational(√5) // 10, 1 // 2 + Rational(√5) // 10, 1]
    b = [1 // 12, 5 // 12, 5 // 12, 1 // 12]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [
        0.08854166666666657,
        0.3830261440755047,
        0.0336405225911624,
        -0.005208333333333329,
    ]
    poly_max = 0.012499999999999997
    dn_coeffs = [-24.0,
        53.665631459994984,
        -53.66563145999497,
        24.0,
        0.8]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

function constructLobattoIIIa5(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 5
    a = [0 0 0 0 0
         (119 + 3 * Rational(√21))//1960 (343 - 9 * Rational(√21))//2520 (392 - 96 * Rational(√21))//2205 (343 - 69 * Rational(√21))//2520 (-21 + 3 * Rational(√21))//1960
         13//320 (392 + 105 * Rational(√21))//2880 8//45 (392 - 105 * Rational(√21))//2880 3//320
         (119 - 3 * Rational(√21))//1960 (343 + 69 * Rational(√21))//2520 (392 + 96 * Rational(√21))//2205 (343 + 9 * Rational(√21))//2520 (-21 - 3 * Rational(√21))//1960
         1//20 49//180 16//45 49//180 1//20]
    c = [0, 1 // 2 - Rational(√21) // 14, 1 // 2, 1 // 2 + Rational(√21) // 14, 1]
    b = [1 // 20, 49 // 180, 16 // 45, 49 // 180, 1 // 20]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [
        0.04062499999999983,
        0.30318418332304287,
        0.17777777777777767,
        -0.030961961100820418,
        0.009374999999999994,
    ]
    poly_max = 0.0029409142833778648
    dn_coeffs = [120.0,
        -280.0,
        320.0,
        -280.0,
        120.0,
        0.3571428571428581]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

# LobattoIIIb
for order in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIb$(order)")
    f = Symbol("constructLobattoIIIb$(order)")
    @eval constructRK(_alg::$(alg), ::Type{T}) where {T} = $(f)(T, _alg.nested_nlsolve)
end

function constructLobattoIIIb2(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 2
    a = [1//2 0
         1//2 0]
    c = [0, 1]
    b = [1 // 2, 1 // 2]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [3 // 8, 1 // 8]
    poly_max = 0.25
    dn_coeffs = [-1, 1, 1]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

function constructLobattoIIIb3(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 3
    a = [1//6 -1//6 0
         1//6 1//3 0
         1//6 5//6 0]
    c = [0, 1 // 2, 1]
    b = [1 // 6, 2 // 3, 1 // 6]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [0.20833333333333337, 0.33333333333333337, -0.04166666666666667]
    poly_max = 0.048112522432468816
    dn_coeffs = [6, -12, 6, 1.5]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

function constructLobattoIIIb4(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 4
    a = [1//12 (-1 - Rational(√5))//24 (-1 + Rational(√5))//24 0
         1//12 (25 + Rational(√5))//120 (25 - 13 * Rational(√5))//120 0
         1//12 (25 + 13 * Rational(√5))//120 (25 - Rational(√5))//120 0
         1//12 (11 - Rational(√5))//24 (11 + Rational(√5))//24 0]
    c = [0, 1 // 2 - Rational(√5) // 10, 1 // 2 + Rational(√5) // 10, 1]
    b = [1 // 12, 5 // 12, 5 // 12, 1 // 12]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [
        0.08854166666666657,
        0.3830261440755047,
        0.0336405225911624,
        -0.005208333333333329,
    ]
    poly_max = 0.012499999999999997
    dn_coeffs = [-24.0,
        53.665631459994984,
        -53.66563145999497,
        24.0,
        0.8]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end

function constructLobattoIIIb5(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 5
    a = [1//20 (-7 - Rational(√21))//120 1//15 (-7 + Rational(√21))//120 0
         1//20 (343 + 9 * Rational(√21))//2520 (56 - 15 * Rational(√21))//315 (343 - 69 * Rational(√21))//2520 0
         1//20 (49 + 12 * Rational(√21))//360 8//45 (49 - 12 * Rational(√21))//360 0
         1//20 (343 + 69 * Rational(√21))//2520 (56 + 15 * Rational(√21))//315 (343 - 9 * Rational(√21))//2520 0
         1//20 (119 - 3 * Rational(√21))//360 13//45 (119 + 3 * Rational(√21))//360 0]
    c = [0, 1 // 2 - Rational(√21) // 14, 1 // 2, 1 // 2 + Rational(√21) // 14, 1]
    b = [1 // 20, 49 // 180, 16 // 45, 49 // 180, 1 // 20]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [
        0.04062499999999983,
        0.30318418332304287,
        0.17777777777777767,
        -0.030961961100820418,
        0.009374999999999994,
    ]

    poly_max = 0.0029409142833778648
    dn_coeffs = [120,
        -280.0,
        320.0,
        -280.0,
        120.0,
        0.3571428571428581]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), t.(dn_coeffs))
    return TU, ITU
end
