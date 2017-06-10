using Base.Test
import BoundaryValueDiffEq:BVPSystem, eval_fun!, Φ!, MIRK_scheme

function func!(x, y, out)
    out[1] = y[2]
    out[2] = 0
end

function boundary!(residual, ua, ub)
    residual[1] = ua[1]-5
    residual[2] = ub[1]
end

n=50
y = vcat(collect(linspace(5,0,n))', zeros(n)')
S = BVPSystem(func!, boundary!, collect(linspace(0,5,n)),
              y, 4)

sol = MIRK_scheme(S)
@test sol.f_converged && norm(diff(diff(S.y[1, :]))) < 1e-10
