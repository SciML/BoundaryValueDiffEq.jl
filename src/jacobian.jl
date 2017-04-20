# g(x) = [x[2], -exp(x[1])]
# y = rand(5,2)
# mapslices(g, y, 2)
# [ForwardDiff.jacobian(g, view(y, i, :)) for i in 1:size(y,1)]

# Computing df/dy
function fun_jac!(fun, x, y, dfdy)
    fy = z -> fun(x, z) # z is a dummy variable
    for i in 1:size(y,1)
        ForwardDiff.jacobian!(view(dfdy,:,:,i), fy, view(y,i,:))
    end
end

# Computing df/dy, df/dp
function fun_jac!(fun, x, y, p, dfdy, dfdp)
    fun_jac!(fun, x, y, dfdy)
    fy = z -> fun(x,z)
    for i in 1:size(y,1)
        fp = z->fy(y, z) # z is a dummy variable
        ForwardDiff.jacobian!(view(dfdp,:,:,i), fp, p)
    end
end

# Computing dbc/da, dbc/db
function bc_jac!(bc, a, b, dbcda, dbcdb)
    fa = z -> bc(z, b)
    fb = z -> bc(a, z) # z is a dummy variable
    ForwardDiff.jacobian!(dbcda, fa, a) #, conf)
    ForwardDiff.jacobian!(dbcdb, fb, b) #, conf)
end
    
# Computing dbc/da, dbc/db and dbc/dp
function bc_jac!(bc, a, b, p, dbcda, dbcdb, dbcdp)
    bc_jac!(bc, a, b, dbcda, dbcdb)
    fp = z -> fun(x, y, z)
    ForwardDiff.jacobian!(dbcdp, fp, p)
end

#    n : int
#        Number of equations in the ODE system.
#    m : int
#        Number of nodes in the mesh.
#    k : int
#        Number of the unknown parameters.

# len = (((m - 1) * n) + length(((m-1)*n+1):(m*n+k)))*2n + (((m - 1) * n) + length(((m-1)*n+1):(m*n+k)))*k
function construct_jacobian_indices!(i_ind, j_ind, n, m, k)
    range1 = 1:((m - 1) * n)
    end1   = length(range1)*2n
    range2 = ((m-1)*n+1):(m*n+k)
    end2   = end1+length(range2)*2n
    end3   = end2+length(range1)*k
    range3 = (m*n+1):(m*n+k)
    i_ind[1:end1] = repeat(range1, inner=n, outer=2)
    i_ind[(end1+1):end2] = repeat(range2, inner=n, outer=2)
    i_ind[(end2+1):end3] = repeat(range1, inner=k)
    i_ind[(end3+1):end]  = repeat(range2, inner=k)

    end1j = length(range1)*n
    end2j = end1+length(range2)*n
    j_ind[1:end1j] = repeat(0:n-1, outer=n*(m-1)) + repeat((0:(m-2))*n, inner=n^2) + 1
    j_ind[(end1j+1):end1] = j_ind[1:end1j] + n
    j_ind[(end1+1):end2j] = repeat(1:n, outer=n+k)
    j_ind[(end2j+1):end2] = j_ind[(end1+1):end2j] + (m-1)*n
    j_ind[(end2+1):end3]  = repeat(range3, outer=(m-1)*n)
    j_ind[(end3+1):end]   = repeat(range3, outer=n+k)
    nothing
end
