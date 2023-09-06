using BoundaryValueDiffEq, DiffEqBase, DiffEqDevTools, LinearAlgebra, Test

for order in (2, 3, 4, 5, 6)
    s = Symbol("MIRK$(order)")
    @eval mirk_solver(::Val{$order}) = $(s)()
end

function func_1!(du, u, p, t)
    du[1, 1] = u[1, 2]
    du[1, 2] = 0
end

function boundary!(residual, u, p, t)
    residual[1, 1] = u[1][1, 1] - 5
    residual[1, 2] = u[end][1, 1]
end

tspan = (0.0, 5.0)
u0 = [5.0 -3.5]
prob = BVProblem(func_1!, boundary!, u0, tspan)

@testset "Affineness" begin
    @testset "MIRK$order" for order in (2, 3, 4, 5, 6)
        @time sol = solve(prob, mirk_solver(Val(order)); dt = 0.2)
        @test norm(diff(first.(sol.u)) .+ 0.2, Inf) + abs(sol[1][1] - 5) < 0.01
    end
end
