using BoundaryValueDiffEq
using DiffEqBase, OrdinaryDiffEq, DiffEqDevTools
using Test, LinearAlgebra

println("Shooting method")

function f(du, u, p, t)
    (x, v) = u
    du[1] = v
    return du[2] = -x
end

function bc!(resid, sol, p, t)
    resid[1] = sol[1][1]
    return resid[2] = sol[end][1] - 1
end

tspan = (0.0, 100.0)
u0 = [0.0, 1.0]
bvp = BVProblem(f, bc!, u0, tspan)
resid_f = Array{Float64}(undef, 2)
sol = solve(bvp, Shooting(Tsit5()))
bc!(resid_f, sol, nothing, sol.t)
@test norm(resid_f) < 1e-7

#Test for complex values
u0 = [0.0, 1.0] .+ 1im
bvp = BVProblem(f, bc!, u0, tspan)
resid_f = Array{ComplexF64}(undef, 2)
sol = solve(bvp, Shooting(Tsit5()))
bc!(resid_f, sol, nothing, sol.t)
@test norm(resid_f) < 1e-7
