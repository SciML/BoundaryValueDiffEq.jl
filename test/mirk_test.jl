using Base.Test
using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq
import BoundaryValueDiffEq:BVPSystem, eval_fun!, Φ!, MIRK_scheme

const n=50

# First order test
function func_1!(x, y, out)
    out[1] = y[2]
    out[2] = 0
end

function boundary!(residual, ua, ub)
    residual[1] = ua[1][1]-5
    residual[2] = ub[end][1]
end

y = vcat(collect(linspace(5,0,n))', zeros(n)')
x = collect(linspace(0,5,n))
S1 = BVPSystem(func_1!, boundary!, x, y, 4)

sol_1 = MIRK_scheme(S1)
@test sol_1.f_converged && norm(diff(diff(S1.y[1, :])), Inf) < 1e-10

# Second order linear test
function func_2!(x, y, out)
    out[1] = y[2]
    out[2] = -y[1]
end

S2 = BVPSystem(func_2!, boundary!, x, y, 4)
sol2 = MIRK_scheme(S2)
sol2_analytical(t) = 5*(cos(t) - cot(5)*sin(t))
@test sol2.f_converged && norm(sol2_analytical.( S2.x ) - S2.y[1, :], Inf) < 1e-5

# Second order nonlinear test
function nonlinear_2!(t, y, du)
    du[1] = y[2]
    du[2] = -sin(y[1])
end

S_nonlinear_2 = BVPSystem(nonlinear_2!, boundary!, x, y, 4)
sol_non2 = MIRK_scheme(S_nonlinear_2)

@test sol_non2.f_converged
