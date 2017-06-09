include("/home/arch/GitHub/BoundaryValueDiffEq.jl/src/BoundaryValueDiffEq.jl")
import BoundaryValueDiffEq:BVPSystem, eval_fun!, Φ!, MIRK_scheme
# Testing function for development, please ignore.
function func!(x, y, out)
    out[1] = y[2]
    out[2] = -y[1]
end

function boundary!(residual, ua, ub)
    residual[1, 1] = ua[1]-5
    residual[1, end] = ub[1]-5
end

S = BVPSystem(func!, boundary!, collect(linspace(0,2pi,50)),
              zeros(2,50), 4)

eval_fun!(S)
Φ!(S)
sol = MIRK_scheme(S)
