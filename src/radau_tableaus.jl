# RadauIIa
for order in (1, 3, 5, 9, 13)
    alg = Symbol("RadauIIa$(order)")
    f = Symbol("constructRadauIIa$(order)")
    @eval constructRK(_alg::$(alg), ::Type{T}) where {T} = $(f)(T, _alg.nested_nlsolve)
end

function constructRadauIIa1(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 1
    a = [1]
    c = [1]
    b = [1]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = zeros(s,s)
    τ_star = 0.5

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa3(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 2
    a = [5//12 -1//12
         3//4 1//4]
    c = [1 // 3, 1]
    b = [3 // 4, 1 // 4]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = zeros(s,s)
    τ_star = 0.0

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa5(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 3
    a = [11 // 45-7 * Rational(√6) // 360 37 // 225-169 * Rational(√6) // 1800 -2 // 225+Rational(√6) // 75
         37 // 225+169 * Rational(√6) // 1800 11 // 45+7 * Rational(√6) // 360 -2 // 225-Rational(√6) // 75
         4 // 9-Rational(√6) // 36 4 // 9+Rational(√6) // 36 1//9]
    c = [2 // 5 - Rational(√6) // 10, 2 // 5 + Rational(√6) // 10, 1]
    b = [4 // 9 - Rational(√6) // 36, 4 // 9 + Rational(√6) // 36, 1 // 9]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = zeros(s,s)
    τ_star = 0.0

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa9(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 5
    c = [
        0.5710419611451768219312e-01,
        0.2768430136381238276800e+00,
        0.5835904323689168200567e+00,
        0.8602401356562194478479e+00,
        1.0,
    ]
    c_p = [1 c[1] c[1]^2 c[1]^3 c[1]^4
           1 c[2] c[2]^2 c[2]^3 c[2]^4
           1 c[3] c[3]^2 c[3]^3 c[3]^4
           1 c[4] c[4]^2 c[4]^3 c[4]^4
           1 c[5] c[5]^2 c[5]^3 c[5]^4]

    c_q = [c[1] c[1]^2/2 c[1]^3/3 c[1]^4/4 c[1]^5/5
           c[2] c[2]^2/2 c[2]^3/3 c[2]^4/4 c[2]^5/5
           c[3] c[3]^2/2 c[3]^3/3 c[3]^4/4 c[3]^5/5
           c[4] c[4]^2/2 c[4]^3/3 c[4]^4/4 c[4]^5/5
           c[5] c[5]^2/2 c[5]^3/3 c[5]^4/4 c[5]^5/5]

    a = c_q / c_p
    b = a[5, :]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = zeros(s,s)
    τ_star = 0.0

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa13(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 7
    c = [
        0.2931642715978489197205e-01,
        0.1480785996684842918500e+00,
        0.3369846902811542990971e+00,
        0.5586715187715501320814e+00,
        0.7692338620300545009169e+00,
        0.9269456713197411148519e+00,
        1.0,
    ]
    c_p = [1 c[1] c[1]^2 c[1]^3 c[1]^4 c[1]^5 c[1]^6
           1 c[2] c[2]^2 c[2]^3 c[2]^4 c[2]^5 c[2]^6
           1 c[3] c[3]^2 c[3]^3 c[3]^4 c[3]^5 c[3]^6
           1 c[4] c[4]^2 c[4]^3 c[4]^4 c[4]^5 c[4]^6
           1 c[5] c[5]^2 c[5]^3 c[5]^4 c[5]^5 c[5]^6
           1 c[6] c[6]^2 c[6]^3 c[6]^4 c[6]^5 c[6]^6
           1 c[7] c[7]^2 c[7]^3 c[7]^4 c[7]^5 c[7]^6]

    c_q = [c[1] c[1]^2/2 c[1]^3/3 c[1]^4/4 c[1]^5/5 c[1]^6/6 c[1]^7/7
           c[2] c[2]^2/2 c[2]^3/3 c[2]^4/4 c[2]^5/5 c[2]^6/6 c[2]^7/7
           c[3] c[3]^2/2 c[3]^3/3 c[3]^4/4 c[3]^5/5 c[3]^6/6 c[3]^7/7
           c[4] c[4]^2/2 c[4]^3/3 c[4]^4/4 c[4]^5/5 c[4]^6/6 c[4]^7/7
           c[5] c[5]^2/2 c[5]^3/3 c[5]^4/4 c[5]^5/5 c[5]^6/6 c[5]^7/7
           c[6] c[6]^2/2 c[6]^3/3 c[6]^4/4 c[6]^5/5 c[6]^6/6 c[6]^7/7
           c[7] c[7]^2/2 c[7]^3/3 c[7]^4/4 c[7]^5/5 c[7]^6/6 c[7]^7/7]

    a = c_q / c_p

    b = a[7, :]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = zeros(s,s)
    τ_star = 0.0

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end
