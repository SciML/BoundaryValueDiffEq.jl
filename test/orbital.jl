using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq
using NLsolve, Sundials

myu= 398600.4418E+9
t0=86400* 2.3577475462484435E+04
t1=86400* 2.3577522023524125E+04
domin = (t0,t1)

y0=[-4.7763169762853989E+06, -3.8386398704441520E+05, -5.3500183933132319E+06, -5.5260776801401034E+03, 1.2118371372546346E+03, 4.8474639033861940E+03]
init_val = [-4.7763169762853989E+06,-3.8386398704441520E+05,-5.3500183933132319E+06,7.0526926403748598E+06,-7.9650476230388973E+05,-1.1911128863666430E+06]

function f(t, y, du)
  r=y[1]^2+y[2]^2+y[3]^2

  du[1] = y[4]
  du[2] = y[5]
  du[3] = y[6]
  du[4] = -myu*y[1]/r
  du[5] = -myu*y[2]/r
  du[6] = -myu*y[3]/r

end

function bc!_generator(resid,sol,init_val)
  resid[1] = sol[1][1]   - init_val[1]
  resid[2] = sol[1][2]   - init_val[2]
  resid[3] = sol[1][3]   - init_val[3]
  resid[4] = sol[end][1] - init_val[4]
  resid[5] = sol[end][2] - init_val[5]
  resid[6] = sol[end][3] - init_val[6]
end
cur_bc! = (resid,sol) -> bc!_generator(resid,sol,init_val)

bvp = BVProblem(f, domin, cur_bc!, y0)
resid_f = Array(Float64, 6)
@time sol = solve(bvp, Shooting(DP5()),force_dtmin=true,abstol=1e-13)
@time sol = solve(bvp, Shooting(DP5(),nlsolve=Sundials.kinsol),force_dtmin=true,abstol=1e-13)
cur_bc!(resid_f,sol)

println(resid_f)
println(sol[1])
