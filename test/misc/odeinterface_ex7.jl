using Test, BoundaryValueDiffEq, LinearAlgebra, ODEInterface, Random

# Adaptation of https://github.com/luchr/ODEInterface.jl/blob/958b6023d1dabf775033d0b89c5401b33100bca3/examples/BasicExamples/ex7.jl
function ex7_f!(du, u, p, t)
    ϵ = p[1]
    u₁, λ = u
    du[1] = (sin(t)^2 + λ * sin(t)^4 / u₁) / ϵ
    du[2] = 0
    return nothing
end

function ex7_2pbc1!(resa, ua, p)
    resa[1] = ua[1] - 1
    return nothing
end

function ex7_2pbc2!(resb, ub, p)
    resb[1] = ub[1] - 1
    return nothing
end

u0 = [0.5, 1.0]
p = [0.1]
tspan = (-π / 2, π / 2)

tpprob = TwoPointBVProblem(ex7_f!, (ex7_2pbc1!, ex7_2pbc2!), u0, tspan, p;
    bcresid_prototype = (zeros(1), zeros(1)))

@info "BVPM2"

sol_bvpm2 = solve(tpprob, BVPM2(); dt = π / 20)
@test SciMLBase.successful_retcode(sol_bvpm2)
resid_f = (Array{Float64, 1}(undef, 1), Array{Float64, 1}(undef, 1))
ex7_2pbc1!(resid_f[1], sol_bvpm2(tspan[1]), nothing)
ex7_2pbc2!(resid_f[2], sol_bvpm2(tspan[2]), nothing)
@test norm(resid_f) < 1e-6

function ex7_f2!(du, u, p, t)
    u₁, λ = u
    du[1] = (sin(t)^2 + λ * sin(t)^4 / u₁) / 0.1
    du[2] = 0
    return nothing
end

@info "BVPSOL"

initial_u0 = [sol_bvpm2(t) .+ rand() for t in tspan[1]:(π / 20):tspan[2]]
tpprob = TwoPointBVProblem(ex7_f2!, (ex7_2pbc1!, ex7_2pbc2!), initial_u0, tspan;
    bcresid_prototype = (zeros(1), zeros(1)))

# Just test that it runs. BVPSOL only works with linearly separable BCs.
# TODO: Implement appolo reentry example from ODEInterface.jl
sol_bvpsol = solve(tpprob, BVPSOL(); dt = π / 20)
