# mesh_selector refine nd redistribute according to the initial mesh
function mesh_selector!(
        cache::AscherCache{iip, T}, z, dmz, mesh, mesh_dt, abstol) where {iip, T}
    (; k, ncomp) = cache
    # weights for mesh selection
    cnsts2 = [1.25e-1, 2.604e-3, 8.019e-3, 2.170e-5, 7.453e-5, 5.208e-4, 9.689e-8,
        3.689e-7, 3.100e-6, 2.451e-5, 2.691e-10, 1.120e-9, 1.076e-8, 9.405e-8,
        1.033e-6, 5.097e-13, 2.290e-12, 2.446e-11, 2.331e-10, 2.936e-9, 3.593e-8,
        7.001e-16, 3.363e-15, 3.921e-14, 4.028e-13, 5.646e-12, 7.531e-11, 1.129e-9]

    koff = Int(k * (k + 1) / 2)
    wgtmsh = 10 * cnsts2[koff]
    root = 1 / (k + 1)
    n = length(mesh) - 1
    slope = Vector{T}(undef, n)
    accum = Vector{T}(undef, n + 1)
    d = Vector{T}(undef, ncomp)
    d1 = similar(d)
    d2 = similar(d)
    # the first interval has to be treated separately from the
    # other intervals (generally the solution on the (i-1)st and ith
    # intervals will be used to approximate the needed derivative, but
    # here the 1st and second intervals are used.)
    hiold = mesh_dt[1]
    @views horder(cache, d1, hiold, dmz[1])
    hiold = mesh_dt[2]
    @views horder(cache, d2, hiold, dmz[2])
    oneovh = 2.0 / (mesh[3] - mesh[1])
    slp = @. (abs(d2 - d1) * wgtmsh * oneovh / (abstol * (T(1) + abs(z[1]))))^root
    slope[1] = maximum(slp)
    slphmx = slope[1] * mesh_dt[1]
    accum[2] = slphmx
    iflip = 1

    # go through the remaining intervals generating slope
    # and  accum
    for i in 2:n
        hiold = mesh_dt[i]
        iflip == -1 && @views horder(cache, d1, hiold, dmz[i])
        iflip == 1 && @views horder(cache, d2, hiold, dmz[i])
        oneovh = 2.0 / (mesh[i + 1] - mesh[i - 1])

        # evaluate function to be equidistributed
        slp = @. (abs(d2 - d1) * wgtmsh * oneovh / (abstol * (T(1) + abs(z[i]))))^root
        slope[i] = maximum(slp)

        # accumulate approximate integral of function to be equidistributed
        temp = slope[i] * hiold
        slphmx = max(slphmx, temp)
        accum[i + 1] = accum[i] + temp
        iflip = -iflip
    end

    avrg = accum[n + 1] / n
    # `degequ` is the degree of equidistribution which then is used to
    # determine whether the mesh redistribution is worthwhile or just
    # halving the mesh is enough.
    degequ = avrg / max(slphmx, eps(T))

    # expected n to achieve 0.1x user requested tolerances
    naccum = floor(Int, accum[n + 1] + 1)

    # decide if mesh selection is worthwhile (otherwise, directly halving mesh is enough)
    if (avrg < eps(T)) || (degequ >= 0.5)
        # Just continue with utilizing the halved mesh which was used for error estimation
        return
    else
        redistribute!(cache, length(cache.original_mesh), naccum, slope, accum)
    end
end

function redistribute!(cache::AscherCache{iip, T}, nold::I, naccum::I,
        slope::Vector{T}, accum::Vector{T}) where {iip, T, I <: Integer}
    (; prob, fixpnt, mesh, mesh_dt) = cache
    n::Int = length(slope)
    nmax = copy(n)
    mesh_old = copy(cache.original_mesh)
    # nmx assures mesh has at least half as many subintervals as the
    # previous mesh
    nmx = max(nold + 1, naccum) / 2

    # assures that halving will be possible later
    nmax2 = nmax / 2

    # the mesh is at most halved
    n = min(nmax2, nold, nmx)

    noldp1 = nold + 1
    nfxp1 = length(fixpnt) + 1
    # ensure that fixpnt is included in the new mesh
    (n < nfxp1) && (n = nfxp1)

    # having decided to generate a new mesh with n subintervals we now
    # do so, taking into account that the nfxpnt points in the array
    # fixpnt must be included in the new mesh.
    inn = 1
    accl = T(0)
    lold = 2
    lcarry = 0
    lnew = 0
    resize!(mesh, n + 1)
    mesh[1] = first(prob.tspan)
    mesh[n + 1] = last(prob.tspan)

    for i in 1:nfxp1
        if i !== nfxp1
            for j in lold:noldp1
                lnew = j
                (fixpnt[i] <= mesh_old[j]) && break
            end
            accr = accum[lnew] + (fixpnt[i] - mesh_old[lnew]) * slope[lnew - 1]
            nregn = (accr - accl) / accum[noldp1] * n - T(0.5)
            nregn = min(nregn, n - inn - nfxp1 + i)
            mesh[inn + nregn + 1] = fixpnt[i]
        else
            accr = accum[noldp1]
            lnew = noldp1
            nregn = n - inn
        end
        if nregn !== 0
            temp = accl
            tsum = (accr - accl) / (nregn + 1)
            for j in 1:nregn
                inn = inn + 1
                temp = temp + tsum
                for l in lold:lnew
                    lcarry = l
                    (temp <= accum[l]) && break
                end
                lold = lcarry
                mesh[inn] = mesh_old[lold - 1] + (temp - accum[lold - 1]) / slope[lold - 1]
            end
        end
        inn = inn + 1
        accl = accr
        lold = lnew
    end
    mesh_dt = diff(mesh)
end

function halve_mesh!(cache::AscherCache)
    (; mesh, mesh_dt, valstr) = cache
    n = length(mesh) - 1
    old_mesh = copy(mesh)
    for i in 1:n
        x = mesh[i]
        hd6 = mesh_dt[i] / 6.0
        for j in 1:4
            x = x + hd6
            (j == 3) && (x = x + hd6)
            @views approx(cache, x, valstr[i][j])
        end
    end

    # halve the current mesh
    N = 2 * n
    resize!(mesh, N + 1)
    resize!(mesh_dt, N)
    mesh[1] = old_mesh[1]

    for i in 1:n
        mesh[2i] = (old_mesh[i] + old_mesh[i + 1]) / 2.0
        mesh[2i + 1] = old_mesh[i + 1]
    end
    mesh_dt[1:end] = diff(mesh)[1:end]
end

# determine the error estimate and test to see if the
# error tolerances are satisfied
function error_estimate!(cache::AscherCache)
    (; k, valstr, mesh, mesh_dt, error) = cache
    # weights for extrapolation error estimate
    cnsts1 = [0.25e0, 0.625e-1, 7.2169e-2, 1.8342e-2, 1.9065e-2, 5.8190e-2, 5.4658e-3,
        5.3370e-3, 1.8890e-2, 2.7792e-2, 1.6095e-3, 1.4964e-3, 7.5938e-3, 5.7573e-3,
        1.8342e-2, 4.673e-3, 4.150e-4, 1.919e-3, 1.468e-3, 6.371e-3, 4.610e-3,
        1.342e-4, 1.138e-4, 4.889e-4, 4.177e-4, 1.374e-3, 1.654e-3, 2.863e-3]
    # assign weights for error estimate
    koff = Int(k * (k + 1) / 2)
    wgterr = cnsts1[koff]
    n = length(mesh) - 1

    # error estimates are to be generated and tested
    # to see if the tolerance requirements are satisfied.
    for i in n:-1:1
        # the error estimates are obtained by combining values of the numerical solutions for two meshes.
        # for each value of iback we will consider the two approximation at 2 points in each of
        # the new subintervals. we work backwards through the subinterval so that new values can be stored
        # in valstr in case they prove to be needed later for an error estimate.
        x = mesh[i] + (mesh_dt[i]) * 2.0 / 3.0
        @views approx(cache, x, valstr[i][3])
        error[i] .= wgterr .*
                    abs.(valstr[i][3] .-
                         (isodd(i) ? valstr[Int((i + 1) / 2)][2] : valstr[Int(i / 2)][4]))

        x = mesh[i] + (mesh_dt[i]) / 3.0
        @views approx(cache, x, valstr[i][2])
        error[i] .= error[i] .+
                    wgterr .*
                    abs.(valstr[i][2] .-
                         (isodd(i) ? valstr[Int((i + 1) / 2)][1] : valstr[Int(i / 2)][3]))
    end
    return maximum(reduce(hcat, error), dims = 2)
end

# determine highest order (piecewise constant) derivatives
# of the current collocation solution
function horder(cache::AscherCache, uhigh, hi, dmzi)
    (; ncomp, k, TU) = cache
    (; coef) = TU
    dn = 1.0 / hi^(k - 1)
    uhigh[1:ncomp] .= 0.0

    idmz = 1
    vdmzi = reduce(vcat, dmzi)
    for j in 1:k
        fact = dn * coef[1, j]
        for id in 1:ncomp
            uhigh[id] = uhigh[id] + fact * vdmzi[idmz]
            idmz = idmz + 1
        end
    end
end
