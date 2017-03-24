using BoundaryValueDiffEq
using Base.Test

function f(t, y, du)
  (x, v) = y
  du[1] = v
  du[2] = -x
end

bc = [[0., 1.], [1., 1.]]
domin = (0.,100.)
bvp = BVProblem(f, domin, bc)
@test solve(bvp)[end] ≈ bc[2]

function lorenz(t,u,du)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end

bc_lorenz = [[1.0,0.0,0.0],[-10.,-11.,29.]]

lorenzprob = BVProblem(lorenz, (0.0,1.0), bc_lorenz)
@test norm(solve(lorenzprob)[end] - bc_lorenz[2]) < 0.5
