using BoundaryValueDiffEq

ϵ = 0.1

function test!(f, y, p, t)
    f[1] = y[2]
    f[2] = (y[1] * y[4] - y[3] * y[2]) / ϵ
    f[3] = y[4]
    f[4] = y[5]
    f[5] = y[6]
    f[6] = (-y[3] * y[6] - y[1] * y[2]) / ϵ
end

function bca!(bc, ya, p)
    bc[1] = ya[1] + 1.0
    bc[2] = ya[3]
    bc[3] = ya[4]
end

function bcb!(bc, yb, p)
    bc[1] = yb[1] - 1.0
    bc[2] = yb[3]
    bc[3] = yb[4]
end
u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
tspan = (0.0, 1.0)
prob = TwoPointBVProblem(
    test!, (bca!, bcb!), u0, tspan, bcresid_prototype = (zeros(3), zeros(3)))
sol = solve(prob, MIRK4(), dt = 0.01, abstol = 1e-3)
