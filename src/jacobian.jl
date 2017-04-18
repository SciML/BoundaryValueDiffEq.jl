# Computing df/dy
function fun_jac!(fun, x, y, dfdy)
    fy = z -> fun(x, z) # z is a dummy variable
    conf = ForwardDiff.JacobianConfig{10}(y)
    ForwardDiff.jacobian!(dfdy, fy, y, conf)
end

# Computing df/dy, df/dp
function fun_jac(fun, x, y, p, dfdy, dfdp)
    fp = z->fun(x, y, z) # z is a dummy variable
    # conf = ForwardDiff.JacobianConfig{10}(p)
    # the size of p should be small, so chunk is not needed. 
    fun_jac!(fun, x, y, dfdy)
    ForwardDiff.jacobian!(dfdp, fp, p) #, confp)
end

# Computing dbc/da, dbc/db
function bc_jac!(bc, a, b, dbcda, dbcdb)
    fa = z -> bc(z, b)
    fb = z -> bc(a, z) # z is a dummy variable
    # conf = ForwardDiff.JacobianConfig{10}(a) # dbcda and dbcdb are expected to be the same size.
    # the size of a and b should be small, and chunk is not needed.
    ForwardDiff.jacobian!(dbcda, fa, a) #, conf)
    ForwardDiff.jacobian!(dbcdb, fb, b) #, conf)
end
    
# Computing dbc/da, dbc/db and dbc/dp
function bc_jac!(bc, a, b, p, dbcda, dbcdb, dbcdp)
    bc_jac!(bc, a, b, dbcda, dbcdb)
    fp = z -> fun(x, y, z)
    # conf = ForwardDiff.JacobianConfig{10}
    # the size of p should be small, and chunk is not needed.
    ForwardDiff.jacobian!(dbcdp, fp, p)
end

