function Φ!(cache::AscherCache{iip, T}, z, res, pt::StandardBVProblem) where {iip, T}
    (; f, mesh, mesh_dt, ncomp, ny, bc, k, p, zeta, residual, zval, yval, gval, delz, dmz, deldmz, g, w, v, dmzo, ipvtg, ipvtw, TU) = cache
    (; acol, rho) = TU
    ncy = ncomp + ny
    n = length(mesh) - 1
    Tz = eltype(z)
    dgz = similar(zval)
    df = zeros(T, ncy, ncy)

    temp_rhs = [[Vector{T}(undef, ncy) for _ in 1:k] for _ in 1:n]
    temp_z = [Vector{Tz}(undef, ncomp) for _ in 1:(n + 1)]
    recursive_unflatten!(temp_z, z)
    rhs_bc = similar(zval)

    # zero the matrices to be computed
    fill!.(w, T(0))

    izeta = 1
    izsave = 1
    # set up the linear system of equations
    for i in 1:n
        # construct a block of a and a corresponding piece of rhs
        xii = mesh[i]
        h = mesh_dt[i]
        @views approx(cache, xii, zval)
        # find rhs boundary value
        bc(gval, zval, p, xii)
        # go thru the ncomp collocation equations and side conditions
        # in the i-th subinterval
        while true
            (izeta > ncomp) && break
            (zeta[izeta] > xii + eps(T)) && break
            rhs_bc[izeta] = -gval[izeta]
            # build a row of a corresponding to a boundary point
            @views gderiv(cache, g[i], izeta, zval, dgz, 1, izeta, pt)
            izeta = izeta + 1
        end

        # assemble collocation equations
        for j in 1:k
            hrho = h * rho[j]
            xcol = xii + hrho
            # find rhs values
            @views approx(cache, xcol, zval, yval, dmzo[i][j][1:ncomp])
            uval = vcat(zval, yval)

            @views f(residual, uval, p, xcol)
            dmzo[i][j][(ncomp + 1):ncy] .= T(0)
            temp_rhs[i][j] .= residual .- dmzo[i][j]

            # fill in ncy rows of  w and v
            @views vwblok(
                cache, xcol, hrho, j, w[i], v[i], ipvtw[i], uval, df, acol[:, j], dmzo[i])
        end

        @views gblock!(cache, h, g[i], izeta, w[i], v[i])

        if i >= n
            izsave = izeta
            # build equation for a side condition.
            # other nonlinear case
            zval = __get_value(cache.z[n + 1])
            @views bc(gval, zval, p, mesh[i + 1])
            while true
                (izeta > ncomp) && break
                # find rhs boundary value
                rhs_bc[izeta] = -gval[izeta]
                # build a row of  a  corresponding to a boundary point
                @views gderiv(cache, g[i], izeta + ncomp, zval, dgz, 2, izeta, pt)
                izeta = izeta + 1
            end
        end
    end

    # assembly process completed
    # solve the linear system
    # ABD matrix decomposition
    @views AlmostBlockDiagonals.factor_shift(g, ipvtg, df)

    # perform forward and backward substitution.
    deldmz .= copy(temp_rhs)
    izet = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            delz[i][izet] = rhs_bc[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], delz[i:(i + 1)], deldmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                delz[i + 1][izet] = rhs_bc[izet]
                izet = izet + 1
            end
        end
    end
    # perform forward and backward substitution
    @views AlmostBlockDiagonals.substitution(g, ipvtg, delz)

    # finally find deldmz
    @views dmzsol!(cache, v, delz, deldmz)

    # project current iterate into current pp-space
    dmz .= copy(dmzo)
    izet::Int = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta::Int = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            temp_z[i][izet] = dgz[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], temp_z[i:(i + 1)], dmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                temp_z[i + 1][izet] = dgz[izet]
                izet = izet + 1
            end
        end
    end

    @views AlmostBlockDiagonals.substitution(g, ipvtg, temp_z)

    # finally find dmz
    @views dmzsol!(cache, v, temp_z, dmz)

    temp_z .= temp_z .+ delz
    dmz .= dmz .+ deldmz

    resids = [Vector{T}(undef, ncy) for _ in 1:(n + 1)]
    for (i, item) in enumerate(temp_rhs)
        for (j, col) in enumerate(eachrow(reduce(hcat, item)))
            resids[i][j] = sum(abs2, col)
        end
    end
    recursive_flatten!(z, temp_z)
    residss = [r[1:ncomp] for r in resids]
    recursive_flatten!(res, residss)

    # update z in cache for next iteration
    new_z = __get_value(temp_z)
    copyto!(cache.z, new_z)
    copyto!(cache.dmz, dmz)
end

function Φ!(cache::AscherCache{iip, T}, z, res, pt::TwoPointBVProblem) where {iip, T}
    (; f, mesh, mesh_dt, ncomp, ny, bc, k, p, zeta, bcresid_prototype, residual, zval, yval, gval, delz, dmz, deldmz, g, w, v, dmzo, ipvtg, ipvtw, TU) = cache
    (; acol, rho) = TU
    ncy = ncomp + ny
    n = length(mesh) - 1
    Tz = eltype(z)
    dgz = similar(zval)
    df = zeros(T, ncy, ncy)
    La = length(first(bcresid_prototype))

    temp_rhs = [[Vector{T}(undef, ncy) for _ in 1:k] for _ in 1:n]
    temp_z = [Vector{Tz}(undef, ncomp) for _ in 1:(n + 1)]
    recursive_unflatten!(temp_z, z)
    rhs_bc = similar(zval)

    # zero the matrices to be computed
    fill!.(w, T(0))

    izeta = 1
    izsave = 1
    # set up the linear system of equations
    for i in 1:n
        # construct a block of a and a corresponding piece of rhs
        xii = mesh[i]
        h = mesh_dt[i]
        @views approx(cache, xii, zval)
        # find rhs boundary value
        @views first(bc)(gval[1:La], zval, p)
        @views last(bc)(gval[(La + 1):end], zval, p)
        # go thru the ncomp collocation equations and side conditions
        # in the i-th subinterval
        while true
            (izeta > ncomp) && break
            (zeta[izeta] > xii + eps(T)) && break
            rhs_bc[izeta] = -gval[izeta]
            # build a row of a corresponding to a boundary point
            @views gderiv(cache, g[i], izeta, zval, dgz, 1, izeta, pt)
            izeta = izeta + 1
        end
        #TODO: whether the previous snippet could be available when i==1 for TwoPointBVProblem?
        # assemble collocation equations
        for j in 1:k
            hrho = h * rho[j]
            xcol = xii + hrho
            # find rhs values
            @views approx(cache, xcol, zval, yval, dmzo[i][j][1:ncomp])
            uval = vcat(zval, yval)

            @views f(residual, uval, p, xcol)
            dmzo[i][j][(ncomp + 1):ncy] .= T(0)
            temp_rhs[i][j] .= residual .- dmzo[i][j]

            # fill in ncy rows of  w and v
            @views vwblok(
                cache, xcol, hrho, j, w[i], v[i], ipvtw[i], uval, df, acol[:, j], dmzo[i])
        end

        @views gblock!(cache, h, g[i], izeta, w[i], v[i])

        if i >= n
            izsave = izeta
            # build equation for a side condition.
            # other nonlinear case
            zval = __get_value(cache.z[n + 1])
            @views first(bc)(gval[1:La], zval, p)
            @views last(bc)(gval[(La + 1):end], zval, p)
            while true
                (izeta > ncomp) && break
                # find rhs boundary value
                rhs_bc[izeta] = -gval[izeta]
                # build a row of  a  corresponding to a boundary point
                @views gderiv(cache, g[i], izeta + ncomp, zval, dgz, 2, izeta, pt)
                izeta = izeta + 1
            end
        end
    end

    # assembly process completed
    # solve the linear system
    # matrix decomposition
    @views AlmostBlockDiagonals.factor_shift(g, ipvtg, df)

    # perform forward and backward substitution.
    deldmz .= copy(temp_rhs)
    izet = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            delz[i][izet] = rhs_bc[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], delz[i:(i + 1)], deldmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                delz[i + 1][izet] = rhs_bc[izet]
                izet = izet + 1
            end
        end
    end
    # perform forward and backward substitution
    @views AlmostBlockDiagonals.substitution(g, ipvtg, delz)

    # finally find deldmz
    @views dmzsol!(cache, v, delz, deldmz)

    # project current iterate into current pp-space
    dmz .= copy(dmzo)
    izet::Int = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta::Int = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            temp_z[i][izet] = dgz[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], temp_z[i:(i + 1)], dmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                temp_z[i + 1][izet] = dgz[izet]
                izet = izet + 1
            end
        end
    end

    @views AlmostBlockDiagonals.substitution(g, ipvtg, temp_z)

    # finally find dmz
    @views dmzsol!(cache, v, temp_z, dmz)

    temp_z .= temp_z .+ delz
    dmz .= dmz .+ deldmz

    resids = [Vector{T}(undef, ncy) for _ in 1:(n + 1)]
    for (i, item) in enumerate(temp_rhs)
        for (j, col) in enumerate(eachrow(reduce(hcat, item)))
            resids[i][j] = sum(abs2, col)
        end
    end
    recursive_flatten!(z, temp_z)
    residss = [r[1:ncomp] for r in resids]
    recursive_flatten!(res, residss)

    # update z in cache for next iteration
    new_z = __get_value(temp_z)
    copyto!(cache.z, new_z)
    copyto!(cache.dmz, dmz)
end

@inline __get_value(z::Vector{<:AbstractArray}) = eltype(first(z)) <: ForwardDiff.Dual ?
                                                  [map(x -> x.value, a) for a in z] : z
@inline __get_value(z) = isa(z, ForwardDiff.Dual) ? z.value : z

function Φ(cache::AscherCache{iip, T}, z, pt::StandardBVProblem) where {iip, T}
    (; f, mesh, mesh_dt, ncomp, ny, bc, k, p, zeta, residual, zval, yval, gval, delz, dmz, deldmz, g, w, v, dmzo, ipvtg, ipvtw, TU) = cache
    (; acol, rho) = TU
    ncy = ncomp + ny
    n = length(mesh) - 1
    Tz = eltype(z)
    dgz = similar(zval)
    df = Matrix{T}(undef, ncy, ncy)

    temp_rhs = [[Vector{T}(undef, ncy) for _ in 1:k] for _ in 1:n]
    temp_z = [Vector{Tz}(undef, ncomp) for _ in 1:(n + 1)]
    recursive_unflatten!(temp_z, z)
    rhs_bc = similar(zval)

    # zero the matrices to be computed
    fill!.(w, 0.0)

    izeta = 1
    izsave = 1
    # set up the linear system of equations
    for i in 1:n
        # construct a block of a and a corresponding piece of rhs
        xii = mesh[i]
        h = mesh_dt[i]
        @views approx(cache, xii, zval)
        # find rhs boundary value
        gval = bc(zval, p, xii)
        # go thru the ncomp collocation equations and side conditions
        # in the i-th subinterval
        while true
            (izeta > ncomp) && break
            (zeta[izeta] > xii + eps(T)) && break
            rhs_bc[izeta] = -gval[izeta]
            # build a row of a corresponding to a boundary point
            @views gderiv(cache, g[i], izeta, zval, dgz, 1, izeta, pt)
            izeta = izeta + 1
        end

        # assemble collocation equations
        for j in 1:k
            hrho = h * rho[j]
            xcol = xii + hrho
            # find rhs values
            @views approx(cache, xcol, zval, yval, dmzo[i][j][1:ncomp])
            uval = vcat(zval, yval)

            residual = f(uval, p, xcol)
            dmzo[i][j][(ncomp + 1):ncy] .= 0.0
            temp_rhs[i][j] .= residual .- dmzo[i][j]

            # fill in ncy rows of  w and v
            @views vwblok(
                cache, xcol, hrho, j, w[i], v[i], ipvtw[i], uval, df, acol[:, j], dmzo[i])
        end

        @views gblock!(cache, h, g[i], izeta, w[i], v[i])

        if i >= n
            izsave = izeta
            # build equation for a side condition.
            # other nonlinear case
            zval = __get_value(cache.z[n + 1])
            gval = bc(zval, p, mesh[i + 1])
            while true
                (izeta > ncomp) && break
                # find rhs boundary value
                rhs_bc[izeta] = -gval[izeta]
                # build a row of  a  corresponding to a boundary point
                @views gderiv(cache, g[i], izeta + ncomp, zval, dgz, 2, izeta, pt)
                izeta = izeta + 1
            end
        end
    end

    # assembly process completed
    # solve the linear system
    # matrix decomposition
    @views AlmostBlockDiagonals.factor_shift(g, ipvtg, df)

    # perform forward and backward substitution.
    deldmz .= copy(temp_rhs)
    izet = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            delz[i][izet] = rhs_bc[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], delz[i:(i + 1)], deldmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                delz[i + 1][izet] = rhs_bc[izet]
                izet = izet + 1
            end
        end
    end
    # perform forward and backward substitution
    @views AlmostBlockDiagonals.substitution(g, ipvtg, delz)

    # finally find deldmz
    @views dmzsol!(cache, v, delz, deldmz)

    # project current iterate into current pp-space
    dmz .= copy(dmzo)
    izet::Int = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta::Int = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            temp_z[i][izet] = dgz[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], temp_z[i:(i + 1)], dmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                temp_z[i + 1][izet] = dgz[izet]
                izet = izet + 1
            end
        end
    end

    @views AlmostBlockDiagonals.substitution(g, ipvtg, temp_z)

    # finally find dmz
    @views dmzsol!(cache, v, temp_z, dmz)

    temp_z .= temp_z .+ delz
    dmz .= dmz .+ deldmz

    resids = [Vector{T}(undef, ncy) for _ in 1:(n + 1)]
    for (i, item) in enumerate(temp_rhs)
        for (j, col) in enumerate(eachrow(reduce(hcat, item)))
            resids[i][j] = sum(abs2, col)
        end
    end
    recursive_flatten!(z, temp_z)
    residss = [r[1:ncomp] for r in resids]

    # update z in cache for next iteration
    new_z = __get_value(temp_z)
    copyto!(cache.z, new_z)
    copyto!(cache.dmz, dmz)

    return reduce(vcat, residss)
end

function Φ(cache::AscherCache{iip, T}, z, pt::TwoPointBVProblem) where {iip, T}
    (; f, mesh, mesh_dt, ncomp, ny, bc, k, p, zeta, residual, zval, yval, gval, delz, dmz, deldmz, g, w, v, dmzo, ipvtg, ipvtw, TU) = cache
    (; acol, rho) = TU
    ncy = ncomp + ny
    n = length(mesh) - 1
    Tz = eltype(z)
    dgz = similar(zval)
    df = Matrix{T}(undef, ncy, ncy)

    temp_rhs = [[Vector{T}(undef, ncy) for _ in 1:k] for _ in 1:n]
    temp_z = [Vector{Tz}(undef, ncomp) for _ in 1:(n + 1)]
    recursive_unflatten!(temp_z, z)
    rhs_bc = similar(zval)

    # zero the matrices to be computed
    fill!.(w, 0.0)

    izeta = 1
    izsave = 1
    # set up the linear system of equations
    for i in 1:n
        # construct a block of a and a corresponding piece of rhs
        xii = mesh[i]
        h = mesh_dt[i]
        @views approx(cache, xii, zval)
        # find rhs boundary value
        gvalₐ = first(bc)(zval, p)
        gvalᵦ = last(bc)(zval, p)
        gval = vcat(gvalₐ, gvalᵦ)
        # go thru the ncomp collocation equations and side conditions
        # in the i-th subinterval
        while true
            (izeta > ncomp) && break
            (zeta[izeta] > xii + eps(T)) && break
            rhs_bc[izeta] = -gval[izeta]
            # build a row of a corresponding to a boundary point
            @views gderiv(cache, g[i], izeta, zval, dgz, 1, izeta, pt)
            izeta = izeta + 1
        end

        # assemble collocation equations
        for j in 1:k
            hrho = h * rho[j]
            xcol = xii + hrho
            # find rhs values
            @views approx(cache, xcol, zval, yval, dmzo[i][j][1:ncomp])
            uval = vcat(zval, yval)

            residual = f(uval, p, xcol)
            dmzo[i][j][(ncomp + 1):ncy] .= 0.0
            temp_rhs[i][j] .= residual .- dmzo[i][j]

            # fill in ncy rows of  w and v
            @views vwblok(
                cache, xcol, hrho, j, w[i], v[i], ipvtw[i], uval, df, acol[:, j], dmzo[i])
        end

        @views gblock!(cache, h, g[i], izeta, w[i], v[i])

        if i >= n
            izsave = izeta
            # build equation for a side condition.
            # other nonlinear case
            zval = __get_value(cache.z[n + 1])
            gvalₐ = first(bc)(zval, p)
            gvalᵦ = last(bc)(zval, p)
            gval = vcat(gvalₐ, gvalᵦ)
            while true
                (izeta > ncomp) && break
                # find rhs boundary value
                rhs_bc[izeta] = -gval[izeta]
                # build a row of  a  corresponding to a boundary point
                @views gderiv(cache, g[i], izeta + ncomp, zval, dgz, 2, izeta, pt)
                izeta = izeta + 1
            end
        end
    end

    # assembly process completed
    # solve the linear system
    # matrix decomposition
    @views AlmostBlockDiagonals.factor_shift(g, ipvtg, df)

    # perform forward and backward substitution.
    deldmz .= copy(temp_rhs)
    izet = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            delz[i][izet] = rhs_bc[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], delz[i:(i + 1)], deldmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                delz[i + 1][izet] = rhs_bc[izet]
                izet = izet + 1
            end
        end
    end
    # perform forward and backward substitution
    @views AlmostBlockDiagonals.substitution(g, ipvtg, delz)

    # finally find deldmz
    @views dmzsol!(cache, v, delz, deldmz)

    # project current iterate into current pp-space
    dmz .= copy(dmzo)
    izet::Int = 1
    for i in 1:n
        nrow = g.rows[i]
        izeta::Int = nrow + 1 - ncomp
        (i == n) && (izeta = izsave)
        while true
            (izet == izeta) && break
            temp_z[i][izet] = dgz[izet]
            izet = izet + 1
        end
        h = mesh_dt[i]
        @views gblock!(cache, h, izeta, w[i], temp_z[i:(i + 1)], dmz[i], ipvtw[i])

        if i == n
            while true
                (izet > ncomp) && break
                temp_z[i + 1][izet] = dgz[izet]
                izet = izet + 1
            end
        end
    end

    @views AlmostBlockDiagonals.substitution(g, ipvtg, temp_z)

    # finally find dmz
    @views dmzsol!(cache, v, temp_z, dmz)

    temp_z .= temp_z .+ delz
    dmz .= dmz .+ deldmz

    resids = [Vector{T}(undef, ncy) for _ in 1:(n + 1)]
    for (i, item) in enumerate(temp_rhs)
        for (j, col) in enumerate(eachrow(reduce(hcat, item)))
            resids[i][j] = sum(abs2, col)
        end
    end
    recursive_flatten!(z, temp_z)
    residss = [r[1:ncomp] for r in resids]

    # update z in cache for next iteration
    new_z = __get_value(temp_z)
    copyto!(cache.z, new_z)
    copyto!(cache.dmz, dmz)

    return reduce(vcat, residss)
end

function approx(cache::AscherCache{iip, T}, x, zval) where {iip, T}
    (; k, z, ncomp, dmz, TU, mesh, mesh_dt) = cache
    (; coef) = TU
    n = length(mesh) - 1
    a = Vector{T}(undef, 7)
    i = interval(mesh, x)
    s = (x - mesh[i]) / mesh_dt[i]
    @views rkbas!(s, coef, k, a)
    bm = x - mesh[i]
    if i == n + 1
        zval .= z[n + 1]
        return
    end
    # evaluate z(u(x))
    for jcomp in 1:ncomp
        zsum = sum(a[j] * dmz[i][j][jcomp] for j in 1:k)
        zᵢ = __get_value(z[i][jcomp])
        zsum = zsum * bm + zᵢ
        zval[jcomp] = zsum
    end
end

function approx(cache::AscherCache{iip, T}, x, zval, yval) where {iip, T}
    (; k, z, ncomp, ny, dmz, TU, mesh, mesh_dt) = cache
    (; coef) = TU
    n = length(mesh) - 1
    dm = Vector{T}(undef, 7)
    a = Vector{T}(undef, 7)
    i = interval(mesh, x)
    s = (x - mesh[i]) / mesh_dt[i]
    @views rkbas!(s, coef, k, a, dm)
    bm = x - mesh[i]
    if i == n + 1
        zval .= z[n + 1]
        yval .= 0.0
        for j in 1:k
            yval .= yval .+ dm[j] * dmz[i - 1][j][(ncomp + 1):end]
        end
        return
    end
    # evaluate z(u(x))
    for jcomp in 1:ncomp
        zsum = sum(a[j] * dmz[i][j][jcomp] for j in 1:k)
        zᵢ = __get_value(z[i][jcomp])
        zsum = zsum * bm + zᵢ
        zval[jcomp] = zsum
    end

    # evaluate  y(j) = j-th component of y.
    yval .= 0.0
    for j in 1:k
        yval .= yval .+ dm[j] * dmz[i][j][(ncomp + 1):end]
    end
end

function approx(cache::AscherCache{iip, T}, x, zval, yval, dmval) where {iip, T}
    (; k, z, ncomp, dmz, TU, mesh, mesh_dt) = cache
    (; coef) = TU
    n = length(mesh) - 1
    dm = Vector{T}(undef, 7)
    a = Vector{T}(undef, 7)
    i = interval(mesh, x)
    s = (x - mesh[i]) / mesh_dt[i]
    @views rkbas!(s, coef, k, a, dm)
    bm = x - mesh[i]
    if i == n + 1
        zval .= z[n + 1]
        yval .= 0.0
        for j in 1:k
            yval .= yval .+ dm[j] * dmz[i - 1][j][(ncomp + 1):end]
        end
        dmval .= 0.0
        for j in 1:k
            @. dmval = dmval + dm[j] * dmz[i - 1][j][1:ncomp]
        end
        return
    end
    # evaluate z(u(x))
    for jcomp in 1:ncomp
        zsum = sum(a[j] * dmz[i][j][jcomp] for j in 1:k)
        zᵢ = __get_value(z[i][jcomp])
        zsum = zsum * bm + zᵢ
        zval[jcomp] = zsum
    end

    # evaluate  y(j) = j-th component of y.
    yval .= 0.0
    for j in 1:k
        @. yval = yval + dm[j] * dmz[i][j][(ncomp + 1):end]
    end

    #  evaluate  dmval(j) = mj-th derivative of uj.
    dmval .= 0.0
    for j in 1:k
        @. dmval = dmval + dm[j] * dmz[i][j][1:ncomp]
    end
end

# construct a group of ncomp rows of the matrices wi and
# corrsponding to an interior collocation point
# jj=1...k
function vwblok(cache::AscherCache, xcol, hrho, jj, wi, vi, ipvtw, zyval, df, acol, dmzo)
    (; jac, k, p, ncomp, ny) = cache
    ncy = ncomp + ny
    kdy = k * ncy
    # initialize wi
    i0 = (jj - 1) * ncy
    for id in (i0 + 1):(i0 + ncomp)
        wi[id, id] = 1.0
    end

    # calculuate local basis
    ha = hrho .* acol

    @views jac(df, zyval, p, xcol)
    i1 = i0 + 1
    i2 = i0 + ncy

    # evaluate dmzo=dmzo - df*(zval,yval) once for a new mesh
    dmzo[jj] .= dmzo[jj] .- df * zyval

    # loop over the  ncomp  expressions to be set up for the
    # current collocation point.
    vi[i1:i2, :] .= df[:, 1:ncomp]
    for jcomp in 1:ncomp
        jw = jcomp
        for j in 1:k
            for iw in i1:i2
                wi[iw, jw] = wi[iw, jw] - ha[j] * vi[iw, jcomp]
            end
            jw = jw + ncy
        end
    end

    # the algebraic solution components
    for jcomp in 1:ny
        jd = ncomp + jcomp
        for id in 1:ncy
            wi[i0 + id, i0 + jd] = -df[id, ncomp + jcomp]
        end
    end

    (jj < k) && return

    # decompose the wi block and solve for the ncomp columns of vi
    # do parameter condensation
    @views __factorize!(wi, ipvtw)
    for j in 1:ncomp
        @views __substitute!(wi, ipvtw, vi[:, j])
    end
    return
end

# construct collocation matrix rows according to mode
function gblock!(cache::AscherCache, h, irow, wi, vrhsz, rhsdmz, ipvtw)
    (; TU, k, ncomp) = cache
    (; b) = TU

    rhsz = reduce(vcat, vrhsz)

    # compute local basis
    hb = @. h * b

    # compute the appropriate piece of rhsz
    @views __substitute!(wi, ipvtw, rhsdmz)

    for jcomp in 1:ncomp
        rhsz[irow + jcomp - 1] = sum(hb[j] * rhsdmz[j][jcomp] for j in 1:k)
    end
    recursive_unflatten!(vrhsz, rhsz)
end

function gblock!(cache::AscherCache, h, gi, irow, wi, vi)
    (; TU, k, ncomp, ny) = cache
    (; b) = TU
    ncy = ncomp + ny

    # compute local basis
    hb = @. h * b

    # branch according to mode
    # set right gi-block identity
    gi[irow:(irow + ncomp - 1), 1:ncomp] .= 0.0
    gi[irow:(irow + ncomp - 1), (ncomp + 1):end] .= 0.0
    for j in 1:ncomp
        gi[irow - 1 + j, ncomp + j] = 1.0
    end

    # compute the block gi
    for icomp in 1:ncomp
        ir = irow + icomp
        id = ir - 1
        for jcol in 1:ncomp
            ind = icomp
            rsum = 0.0
            for j in 1:k
                rsum = rsum - hb[j] * vi[ind, jcol]
                ind = ind + ncy
            end
            gi[id, jcol] = rsum
        end
        gi[id, icomp] = gi[id, icomp] - 1.0
    end
end

function dmzsol!(cache::AscherCache, v, z, dmz)
    (; k, ncomp, ny) = cache
    n = length(dmz)
    ncy = ncomp + ny
    kdy = k * ncy
    for i in 1:n
        for j in 1:ncomp
            fact = __get_value(z[i][j])
            for l in 1:kdy
                kk, jj = __locate_stage(l, ncy)
                dmz[i][kk][jj] = dmz[i][kk][jj] + fact * v[i][l, j]
            end
        end
    end
    return nothing
end
@inline function __locate_stage(l, ncy)
    (1 ≤ l ≤ ncy) && (return 1, l)
    (ncy + 1 ≤ l ≤ 2 * ncy) && (return 2, l - ncy)
    (2 * ncy + 1 ≤ l ≤ 3 * ncy) && (return 3, l - 2 * ncy)
    (3 * ncy + 1 ≤ l ≤ 4 * ncy) && (return 4, l - 3 * ncy)
    (4 * ncy + 1 ≤ l ≤ 5 * ncy) && (return 5, l - 4 * ncy)
    (5 * ncy + 1 ≤ l ≤ 6 * ncy) && (return 6, l - 5 * ncy)
    (6 * ncy + 1 ≤ l ≤ 7 * ncy) && (return 7, l - 6 * ncy)
end

function gderiv(cache::AscherCache{iip, T}, gi, irow, zval, dgz,
        mode::Integer, izeta, pt::StandardBVProblem) where {iip, T}
    (; ncomp, bcjac) = cache
    # construct a collocation matrix row according to mode:
    # mode = 1 - a row corresponding to a initial condition
    # mode = 2 - a row corresponding to a condition at aright
    ddg = Matrix{T}(undef, ncomp, ncomp)

    # evaluate boundary conditin jacobian
    @views bcjac(ddg, zval, nothing, nothing)
    dg = ddg[izeta, :]

    # evaluate dgz = dg * zval once for a new mesh
    dgz[izeta] = sum(dg .* zval)

    # branch according to mode
    if mode !== 2
        # provide coefficients of the j-th linearized side condition.
        # specifically, at x=zeta(j) the j-th side condition reads
        # dg(1)*z(1) + ... +dg(ncomp)*z(ncomp) + g = 0

        # handle an initial condition
        gi[irow, 1:ncomp] .= dg
        gi[irow, (ncomp + 1):end] .= 0.0
    else
        # handle a final condition
        gi[irow, 1:ncomp] .= 0.0
        gi[irow, (ncomp + 1):end] .= dg
    end
end

function gderiv(cache::AscherCache{iip, T}, gi, irow, zval, dgz,
        mode::Integer, izeta, pt::TwoPointBVProblem) where {iip, T}
    (; ncomp, bcjac) = cache
    # construct a collocation matrix row according to mode:
    # mode = 1 - a row corresponding to a initial condition
    # mode = 2 - a row corresponding to a condition at aright
    ddg = Matrix{T}(undef, ncomp, ncomp)

    # evaluate boundary conditin jacobian
    @views bcjac(ddg, zval, nothing)
    dg = ddg[izeta, :]

    # evaluate dgz = dg * zval once for a new mesh
    dgz[izeta] = sum(dg .* zval)

    # branch according to mode
    if mode !== 2
        # provide coefficients of the j-th linearized side condition.
        # specifically, at x=zeta(j) the j-th side condition reads
        # dg(1)*z(1) + ... +dg(ncomp)*z(ncomp) + g = 0

        # handle an initial condition
        gi[irow, 1:ncomp] .= dg
        gi[irow, (ncomp + 1):end] .= 0.0
    else
        # handle a final condition
        gi[irow, 1:ncomp] .= 0.0
        gi[irow, (ncomp + 1):end] .= dg
    end
end

function interval(mesh, t)
    a = findfirst(x -> x ≈ t, mesh)
    # CODLAE actually evaluate the value at final mesh point at mesh[n]
    (a == length(mesh)) && (return length(mesh) - 1)
    n = length(mesh)
    a === nothing ? (return clamp(searchsortedfirst(mesh, t) - 1, 1, n)) : a
end
