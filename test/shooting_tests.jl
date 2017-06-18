using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq, DiffEqDevTools
using Base.Test

println("Shooting method")

function f(t, y, du)
  (x, v) = y
  du[1] = v
  du[2] = -x
end

function bc!(resid,sol)
  resid[1] = sol[1][1]
  resid[2] = sol[end][1] - 1
end

tspan = (0.,100.)
u0 = [0.,1.]
bvp = BVProblem(f, bc!, u0, tspan)
resid_f = Array{Float64}(2)
bc!(resid_f, solve(bvp, Shooting(Tsit5())))
@test norm(resid_f) < 1e-7

