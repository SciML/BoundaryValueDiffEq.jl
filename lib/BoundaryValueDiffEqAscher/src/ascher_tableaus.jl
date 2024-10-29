function constructAscher(alg::Ascher1, ::Type{T}) where {T}
    # initialization
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    acol = Matrix{T}(undef, k, k)

    # collocation points
    rho = [1 // 2]
    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    b = [1]
    return AscherTableau(rho, coef, b, acol)
end
function constructAscher(alg::Ascher2, ::Type{T}) where {T}
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    acol = Matrix{T}(undef, k, k)

    # Gauss Legendre collocation points
    rho = [1 // 2 - Rational(sqrt(3)) // 6, 1 // 2 + Rational(sqrt(3)) // 6]

    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    b = [1 // 2, 1 // 2]
    return AscherTableau(rho, coef, b, acol)
end
function constructAscher(alg::Ascher3, ::Type{T}) where {T}
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    acol = Matrix{T}(undef, k, k)

    # Gauss Legendre collocation points
    rho = [1 // 2 - Rational(sqrt(15)) // 10, 1 // 2, 1 // 2 + Rational(sqrt(15)) // 10]

    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    b = [5 // 18, 4 // 9, 5 // 18]
    return AscherTableau(rho, coef, b, acol)
end
function constructAscher(alg::Ascher4, ::Type{T}) where {T}
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    b = Vector{T}(undef, k)
    acol = Matrix{T}(undef, k, k)

    # Gauss Legendre collocation points
    rho = [0.06943184420297371373, 0.33000947820757187134,
        0.66999052179242812866, 0.93056815579702628627]

    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    @views rkbas!(1.0, coef, k, b)
    return AscherTableau(rho, coef, b, acol)
end
function constructAscher(alg::Ascher5, ::Type{T}) where {T}
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    b = Vector{T}(undef, k)
    acol = Matrix{T}(undef, k, k)

    # Gauss Legendre collocation points
    rho = [0.04691007703066801815, 0.23076534494715844614, 0.5,
        0.76923465505284155386, 0.95308992296933198185]

    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    @views rkbas!(1.0, coef, k, b)
    return AscherTableau(rho, coef, b, acol)
end
function constructAscher(alg::Ascher6, ::Type{T}) where {T}
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    b = Vector{T}(undef, k)
    acol = Matrix{T}(undef, k, k)

    # Gauss Legendre collocation points
    rho = [0.03376524289842397497, 0.16939530676686775923, 0.38069040695840154764,
        0.61930959304159845236, 0.83060469323313224077, 0.96623475710157602503]

    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    @views rkbas!(1.0, coef, k, b)
    return AscherTableau(rho, coef, b, acol)
end
function constructAscher(alg::Ascher7, ::Type{T}) where {T}
    k = alg_stage(alg)
    coef = zeros(k, k) + I
    b = Vector{T}(undef, k)
    acol = Matrix{T}(undef, k, k)

    # Gauss Legendre collocation points
    rho = [0.02544600438286209743, 0.12923440720030276996, 0.29707742431130140792, 0.5,
        0.70292257568869859208, 0.87076559279969723004, 0.97455399561713790257]
    # find runge-kutta coefficients b, acol
    for j in 1:k
        @views vmonde!(rho, coef[:, j], k)
    end
    for i in 1:k
        @views rkbas!(rho[i], coef, k, acol[:, i])
    end
    @views rkbas!(1.0, coef, k, b)
    return AscherTableau(rho, coef, b, acol)
end

# Solve vandermonde system v*x=e
# with v(i,j)=rho(j)**(i-1)/(i-1)!
function vmonde!(rho, coef, k)
    (k == 1) && return
    for i in 1:(k - 1)
        for j in 1:(k - i)
            coef[j] = (coef[j + 1] - coef[j]) / (rho[j + i] - rho[j])
        end
    end
    ifac = 1
    for i in 1:(k - 1)
        kmi = k + 1 - i
        for j in 2:kmi
            coef[j] = coef[j] - rho[j + i - 1] * coef[j - 1]
        end
        coef[kmi] = ifac * coef[kmi]
        ifac = ifac * i
    end
    coef[1] = ifac * coef[1]
end

function rkbas!(s, coef, k::Integer, rkb, dm)
    # for more guass collocation points per subinterval
    if k == 1
        rkb[1] = 1.0
        dm[1] = 1.0
        return
    end
    t = s ./ (1:k)
    for i in 1:k
        p = coef[1, i]
        for j in 2:k
            p = p * t[k + 2 - j] + coef[j, i]
        end
        rkb[i] = p
    end
    for i in 1:k
        p = coef[1, i]
        for j in 2:k
            p = p * t[k + 1 - j] + coef[j, i]
        end
        dm[i] = p
    end
end

function rkbas!(s, coef, k::Integer, rkb)
    # for more guass collocation points per subinterval
    if k == 1
        rkb[1] = 1.0
        return
    end
    t = s ./ (1:k)
    for i in 1:k
        p = coef[1, i]
        for j in 2:k
            p = p * t[k + 2 - j] + coef[j, i]
        end
        rkb[i] = p
    end
end
