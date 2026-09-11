# Shared problem set and stage-dispatch helpers for the expanded-formulation solver
# tests. Included by firk_basic_tests.jl, firk_affineness_tests.jl and
# firk_convergence_tests.jl, which run as separate test groups.

using SciMLBase: BVProblem, ODEFunction, TwoPointBVProblem, solve

nested = false

for stage in (2, 3, 4, 5)
    s = Symbol("LobattoIIIa$(stage)")
    @eval lobattoIIIa_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

for stage in (2, 3, 4, 5)
    s = Symbol("LobattoIIIb$(stage)")
    @eval lobattoIIIb_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

for stage in (2, 3, 4, 5)
    s = Symbol("LobattoIIIc$(stage)")
    @eval lobattoIIIc_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

for stage in (2, 3, 5, 7)
    s = Symbol("RadauIIa$(stage)")
    @eval radau_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
end

# First order test
function f1!(du, u, p, t)
    du[1] = u[2]
    return du[2] = 0
end
f1(u, p, t) = [u[2], 0]

# Second order linear test
function f2!(du, u, p, t)
    du[1] = u[2]
    return du[2] = -u[1]
end
f2(u, p, t) = [u[2], -u[1]]

function boundary!(residual, u, p, t)
    residual[1] = u(0.0)[1] - 5
    return residual[2] = u(5.0)[1]
end
boundary(u, p, t) = [u(0.0)[1] - 5, u(5.0)[1]]

# Array indexing for boundary conditions
function boundary_indexing!(residual, u, p, t)
    residual[1] = u[:, 1][1] - 5
    return residual[2] = u[:, end][1]
end
boundary_indexing(u, p, t) = [u[:, 1][1] - 5, u[:, end][1]]

function boundary_two_point_a!(resida, ua, p)
    return resida[1] = ua[1] - 5
end
function boundary_two_point_b!(residb, ub, p)
    return residb[1] = ub[1]
end

boundary_two_point_a(ua, p) = [ua[1] - 5]
boundary_two_point_b(ub, p) = [ub[1]]

# Not able to change the initial condition.
# Hard coded solution.
odef1! = ODEFunction(f1!, analytic = (u0, p, t) -> [5 - t, -1])
odef1 = ODEFunction(f1, analytic = (u0, p, t) -> [5 - t, -1])

odef2! = ODEFunction(
    f2!, analytic = (
        u0, p, t,
    ) -> [5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))]
)
odef2 = ODEFunction(
    f2, analytic = (
        u0, p, t,
    ) -> [5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))]
)

bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

tspan = (0.0, 5.0)
u0 = [5.0, -3.5]

probArr = [
    BVProblem(odef1!, boundary!, u0, tspan, nlls = Val(false)),
    BVProblem(odef1, boundary, u0, tspan, nlls = Val(false)),
    BVProblem(odef2!, boundary!, u0, tspan, nlls = Val(false)),
    BVProblem(odef2, boundary, u0, tspan, nlls = Val(false)),
    BVProblem(odef2!, boundary_indexing!, u0, tspan, nlls = Val(false)),
    BVProblem(odef2, boundary_indexing, u0, tspan, nlls = Val(false)),
    TwoPointBVProblem(
        odef1!, (boundary_two_point_a!, boundary_two_point_b!),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
    TwoPointBVProblem(
        odef1, (boundary_two_point_a, boundary_two_point_b),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
    TwoPointBVProblem(
        odef2!, (boundary_two_point_a!, boundary_two_point_b!),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
    TwoPointBVProblem(
        odef2, (boundary_two_point_a, boundary_two_point_b),
        u0, tspan; bcresid_prototype, nlls = Val(false)
    ),
]

testTol = 0.3
affineTol = 1.0e-2
dts = 1 .// 2 .^ (5:-1:3)
# Stage-4 Lobatto methods reach ~1e-14 error (the double-precision floor) at the
# finest `dts` step, which corrupts the Richardson order estimate. A coarser step
# range keeps the finest error above the round-off floor (~1e-12) so the measured
# order reflects the true order 6 rather than floating-point noise.
dts_stage4 = 1 .// 2 .^ (4:-1:2)
