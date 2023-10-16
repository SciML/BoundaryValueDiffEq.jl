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

    # Interpolant coefficients and p(x) max
    poly_coeffs = [1 // 2]
    poly_max = 1.0
    dn_coeffs = [1.0]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), T.(dn_coeffs), Int64(stage), nested)
    return TU, ITU
end

function constructRadauIIa3(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 2
    a = [5//12 -1//12
         3//4 1//4]
    c = [1 // 3, 1]
    b = [3 // 4, 1 // 4]

    # Interpolant coefficients and p(x) max
    poly_coeffs = [0.5625, -0.0625]
    poly_max = 1 // 3
    dn_coeffs = [-2, 2, 1.3333333333333335]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), T.(dn_coeffs), Int64(stage), nested)
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

    # Interpolant coefficients and p(x) max
    poly_coeffs = [0.382961306940849, 0.14481647083692872, -0.027777777777777735]
    poly_max = 0.1
    dn_coeffs = [4.3484692283495345, -10.348469228349535, 6.0, 0.9]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), T.(dn_coeffs), Int64(stage), nested)
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

    # Interpolant coefficients and p(x) max
    poly_coeffs = [
        0.14162553295705615,
        0.2899064921881931,
        0.08419708339605547,
        -0.023229108541305443,
        0.0075,
    ]
    poly_max = 0.007936507936507936
    dn_coeffs = [54.35432870991608,
        -167.45423544989396,
        255.9539629158005,
        -262.8540561758225,
        120.0,
        0.19841269841269948]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), T.(dn_coeffs), Int64(stage), nested)
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

    # Interpolant coefficients and p(x) max
    poly_coeffs = [
        0.07525040363897162,
        0.1560619574068569,
        0.22009145086760462,
        0.05944815647539037,
        -0.01646794001947477,
        0.00880474714086077,
        -0.0031887755102048693,
    ]
    poly_max = 0.0005827505827505828
    dn_coeffs = [1648.7143992159574,
        -5415.177583593382,
        9437.603481951755,
        -12468.061993282445,
        13504.443508516517,
        -11747.521812808422,
        5040.0,
        0.028554778554777505]

    TU = RKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = RKInterpTableau(T.(poly_coeffs), T.(poly_max), T.(dn_coeffs), Int64(stage), nested)
    return TU, ITU
end
