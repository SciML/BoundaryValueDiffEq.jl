using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq, DiffEqDevTools
using Test, LinearAlgebra

println("Shooting method")

function f(du,u,p,t)
  (x, v) = u
  du[1] = v
  du[2] = -x
end

function bc!(resid,sol,p,t)
  resid[1] = sol[1][1]
  resid[2] = sol[end][1] - 1
end

tspan = (0.,100.)
u0 = [0.,1.]
bvp = BVProblem(f, bc!, u0, tspan)
resid_f = Array{Float64}(undef, 2)
sol = solve(bvp, Shooting(Tsit5()))
bc!(resid_f, sol, nothing, sol.t)
@test norm(resid_f) < 1e-7
