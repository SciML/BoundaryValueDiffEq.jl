# RadauIIa
for stage in (1, 2, 3, 5, 7)
    alg = Symbol("RadauIIa$(stage)")
    f = Symbol("constructRadauIIa$(stage)")
    @eval constructRK(_alg::$(alg), ::Type{T}) where {T} = $(f)(T, _alg.nested_nlsolve)
end

function constructRadauIIa1(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 1
    a = [1]
    c = [1]
    b = [1]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = [1.0;;]
    τ_star = 0.0

    TU = FIRKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = FIRKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa2(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 2
    a = [5//12 -1//12
         3//4 1//4]
    a = permutedims(a, (2, 1))
    c = [1 // 3, 1]
    b = [3 // 4, 1 // 4]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = [1.5 -0.5;
               -0.75 0.75]
    τ_star = 0.0

    TU = FIRKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = FIRKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa3(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 3
    a = [11 // 45-7 * Rational(√6) // 360 37 // 225-169 * Rational(√6) // 1800 -2 // 225+Rational(√6) // 75
         37 // 225+169 * Rational(√6) // 1800 11 // 45+7 * Rational(√6) // 360 -2 // 225-Rational(√6) // 75
         4 // 9-Rational(√6) // 36 4 // 9+Rational(√6) // 36 1//9]
    a = permutedims(a, (2, 1))
    c = [2 // 5 - Rational(√6) // 10, 2 // 5 + Rational(√6) // 10, 1]
    b = [4 // 9 - Rational(√6) // 36, 4 // 9 + Rational(√6) // 36, 1 // 9]

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = [1.5580782047249224 -0.8914115380582555 0.33333333333333315;
               -1.9869472213484427 3.320280554681775 -1.3333333333333326;
               0.8052720793239877 -1.9163831904350983 1.1111111111111107]
    τ_star = 0.0

    TU = FIRKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = FIRKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa5(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 5
    c = [0.5710419611451768219312e-01, 0.2768430136381238276800e+00,
        0.5835904323689168200567e+00, 0.8602401356562194478479e+00, 1.0]
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
    a = permutedims(a, (2, 1))

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = [1.5864079001863276 -1.0081178814983707 0.7309748661597844 -0.5092648848477398 0.19999999999999882;
               -5.939631780296778 10.780734269705029 -8.510911966412747 6.069809477004479 -2.3999999999999866;
               9.977775909015945 -24.44476872321262 26.826271868712684 -20.759279054515957 8.399999999999956;
               -7.7637202739307325 21.986586239050933 -29.484574687947546 26.461708722827275 -11.19999999999994;
               2.282881805816463 -7.033077888895508 10.750066442463563 -11.039870359384485 5.0399999999999725]
    τ_star = 0.0

    TU = FIRKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = FIRKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end

function constructRadauIIa7(::Type{T}, nested::Bool) where {T}
    # RK coefficients tableau
    s = 7
    c = [0.2931642715978489197205e-01, 0.1480785996684842918500e+00,
        0.3369846902811542990971e+00, 0.5586715187715501320814e+00,
        0.7692338620300545009169e+00, 0.9269456713197411148519e+00, 1.0]
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

    a = permutedims(a, (2, 1))

    # Coefficients for constructing q and zeros of p(x) polynomial in bvp5c paper
    q_coeff = [1.5940642185610567 -1.036553752196515 0.79382172349084 -0.6325776522499784 0.4976107136030369 -0.3592223940655934 0.14285714285715354;
               -11.867354907681566 21.895554926684994 -18.27080167953177 14.932007947071362 -11.86801681989069 8.60718196191934 -3.4285714285716815;
               42.56843764866437 -104.58826140801058 119.44339657888817 -106.16674936195061 87.24612664353391 -64.21723581541276 25.714285714287506;
               -82.61199090291213 232.1402530759895 -317.623307111846 322.4188516023496 -280.67924303496534 212.06972208567558 -85.71428571429115;
               88.92081439942109 -269.58788741224174 412.88176210349144 -468.1955191607566 439.58250988570325 -345.03025124419685 141.42857142857926;
               -49.9855578088297 158.9633239184469 -262.58964790376285 324.5017915419607 -328.4240528650557 270.6770002601034 -113.14285714286237;
               11.456081588332877 -37.62732723293888 65.57712817877311 -86.63425000191717 93.83554041389372 -81.6275811094104 35.020408163266595]
    τ_star = 0.0

    TU = FIRKTableau(Int64(s), T.(a), T.(c), T.(b), nested)
    ITU = FIRKInterpTableau(T.(q_coeff), T.(τ_star), Int64(s), nested)
    return TU, ITU
end
