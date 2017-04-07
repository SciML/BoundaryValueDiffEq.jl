using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq
using Base.Test

function f(t, y, du)
  (x, v) = y
  du[1] = v
  du[2] = -x
end

function bc!(resid,sol)
  resid[1] = sol[1][1]
  resid[2] = sol[end][1] - 1
end

domin = (0.,100.)
init = [0.,1.]
bvp = BVProblem(f, domin, bc!, init)
resid_f = Array(Float64, 2)
bc!(resid_f, solve(bvp, Shooting(Tsit5())))
@test norm(resid_f) < 1e-7

# function lorenz(t,u,du)
#  du[1] = 10.0(u[2]-u[1])
#  du[2] = u[1]*(28.0-u[3]) - u[2]
#  du[3] = u[1]*u[2] - (8/3)*u[3]
# end

# bc_lorenz(sol) = [sol[1][1]-1, sol[end][1]+10]
# init_lorenz = [1.0,0.0,0.0]
# lorenzprob = BVProblem(lorenz, (0.0,1.0), bc_lorenz, init_lorenz)
# @test norm(bc_lorenz(solve(lorenzprob,Shooting(Tsit5())))) < 0.7
