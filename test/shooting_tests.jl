using BoundaryValueDiffEq, LinearAlgebra, OrdinaryDiffEq, Test

@info "Shooting method"

@info "Multi-Point BVProblem" # Not really but using that API

tspan = (0.0, 100.0)
u0 = [0.0, 1.0]

# Inplace
function f1!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end

function bc1!(resid, sol, p, t)
    t₀, t₁ = first(t), last(t)
    resid[1] = sol(t₀)[1]
    resid[2] = sol(t₁)[1] - 1
    return nothing
end

bvp1 = BVProblem(f1!, bc1!, u0, tspan)
@test SciMLBase.isinplace(bvp1)
resid_f = Array{Float64}(undef, 2)
sol = solve(bvp1, Shooting(Tsit5()); abstol = 1e-6, reltol = 1e-6)
@test SciMLBase.successful_retcode(sol)
bc1!(resid_f, sol, nothing, sol.t)
@test norm(resid_f) < 1e-6

# Out of Place
f1(u, p, t) = [u[2], -u[1]]

function bc1(sol, p, t)
    t₀, t₁ = first(t), last(t)
    return [sol(t₀)[1], sol(t₁)[1] - 1]
end

@test_throws SciMLBase.NonconformingFunctionsError BVProblem(f1!, bc1, u0, tspan)
@test_throws SciMLBase.NonconformingFunctionsError BVProblem(f1, bc1!, u0, tspan)

bvp2 = BVProblem(f1, bc1, u0, tspan)
@test !SciMLBase.isinplace(bvp2)
sol = solve(bvp2, Shooting(Tsit5()); abstol = 1e-6, reltol = 1e-6)
@test SciMLBase.successful_retcode(sol)
resid_f = bc1(sol, nothing, sol.t)
@test norm(resid_f) < 1e-6

@info "Two Point BVProblem" # Not really but using that API

# Inplace
function bc2!((resida, residb), (ua, ub), p)
    resida[1] = ua[1]
    residb[1] = ub[1] - 1
    return nothing
end

bvp3 = TwoPointBVProblem(f1!, bc2!, u0, tspan;
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1)))
@test SciMLBase.isinplace(bvp3)
sol = solve(bvp3, Shooting(Tsit5()); abstol = 1e-6, reltol = 1e-6)
@test SciMLBase.successful_retcode(sol)
resid_f = (Array{Float64, 1}(undef, 1), Array{Float64, 1}(undef, 1))
bc2!(resid_f, (sol(tspan[1]), sol(tspan[2])), nothing)
@test_broken norm(resid_f) < 1e-6
@test norm(resid_f) < 1e-4

# Out of Place
function bc2((ua, ub), p)
    return ([ua[1]], [ub[1] - 1])
end

bvp4 = TwoPointBVProblem(f1, bc2, u0, tspan)
@test !SciMLBase.isinplace(bvp4)
sol = solve(bvp4, Shooting(Tsit5()); abstol = 1e-6, reltol = 1e-6)
@test SciMLBase.successful_retcode(sol)
resid_f = reduce(vcat, bc2((sol(tspan[1]), sol(tspan[2])), nothing))
@test norm(resid_f) < 1e-6

#Test for complex values
u0 = [0.0, 1.0] .+ 1im
bvp = BVProblem(f1!, bc1!, u0, tspan)
resid_f = Array{ComplexF64}(undef, 2)
sol = solve(bvp, Shooting(Tsit5(); nlsolve = NewtonRaphson(; autodiff = AutoFiniteDiff()));
    abstol = 1e-6, reltol = 1e-6)
resid_f = Array{ComplexF64}(undef, 2)
bc1!(resid_f, sol, nothing, sol.t)
@test norm(resid_f) < 1e-6
