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
    return mesh_dt[1:end] = diff(mesh)[1:end]
end

# determine the error estimate and test to see if the
# error tolerances are satisfied
function error_estimate!(cache::AscherCache)
    (; k, valstr, mesh, mesh_dt, error) = cache
    # weights for extrapolation error estimate
    cnsts1 = [
        0.25e0, 0.625e-1, 7.2169e-2, 1.8342e-2, 1.9065e-2, 5.819e-2, 5.4658e-3,
        5.337e-3, 1.889e-2, 2.7792e-2, 1.6095e-3, 1.4964e-3, 7.5938e-3, 5.7573e-3,
        1.8342e-2, 4.673e-3, 4.15e-4, 1.919e-3, 1.468e-3, 6.371e-3, 4.61e-3,
        1.342e-4, 1.138e-4, 4.889e-4, 4.177e-4, 1.374e-3, 1.654e-3, 2.863e-3,
    ]
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
        error[i] .= wgterr .* abs.(
            valstr[i][3] .-
                (isodd(i) ? valstr[Int((i + 1) / 2)][2] : valstr[Int(i / 2)][4])
        )

        x = mesh[i] + (mesh_dt[i]) / 3.0
        @views approx(cache, x, valstr[i][2])
        error[i] .= error[i] .+
            wgterr .* abs.(
            valstr[i][2] .-
                (isodd(i) ? valstr[Int((i + 1) / 2)][1] : valstr[Int(i / 2)][3])
        )
    end
    return maximum(reduce(hcat, error), dims = 2)
end
