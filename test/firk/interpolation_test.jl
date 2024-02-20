using BoundaryValueDiffEq, DiffEqBase, DiffEqDevTools, LinearAlgebra, Test

λ = 1
function prob_bvp_linear_analytic(u, λ, t)
    a = 1 / sqrt(λ)
    [(exp(-a * t) - exp((t - 2) * a)) / (1 - exp(-2 * a)),
        (-a * exp(-t * a) - a * exp((t - 2) * a)) / (1 - exp(-2 * a))]
end
function prob_bvp_linear_f!(du, u, p, t)
    du[1] = u[2]
    du[2] = 1 / p * u[1]
end
function prob_bvp_linear_bc!(res, u, p, t)
    res[1] = u[1][1] - 1
    res[2] = u[end][1]
end
prob_bvp_linear_function = ODEFunction(prob_bvp_linear_f!,
                                       analytic = prob_bvp_linear_analytic)
prob_bvp_linear_tspan = (0.0, 1.0)
prob_bvp_linear = BVProblem(prob_bvp_linear_function, prob_bvp_linear_bc!,
                            [1.0, 0.0], prob_bvp_linear_tspan, λ)
testTol = 1e-6
nested = true

@testset "Radau interpolations" begin
    for order in (3, 5, 9, 13)
        s = Symbol("RadauIIa$(order)")
        @eval radau_solver(::Val{$order}) = $(s)(NewtonRaphson(),BVPJacobianAlgorithm(AutoFiniteDiff()), nested)
    end
    @testset "Interpolation" begin @testset "RadauIIa$order" for order in (3, 5, 9, 13)
        @time sol = solve(prob_bvp_linear, radau_solver(Val(order)); dt = 0.001)
        @test sol(0.001)≈[0.998687464, -1.312035941] atol=testTol
    end end
end

@testset "LobattoIII interpolations" begin for lobatto in ("a", "b", "c")
    for order in (3, 4, 5)
        s = Symbol("LobattoIII$(lobatto)$(order)")
        @eval lobatto_solver(::Val{$order}) = $(s)(NewtonRaphson(),BVPJacobianAlgorithm(AutoFiniteDiff()), nested)
    end

    @testset "Interpolation" begin @testset "LobattoIII$(lobatto)$order" for order in (3,
                                                                                       4, 5)
        @time sol = solve(prob_bvp_linear, lobatto_solver(Val(order)); dt = 0.001)
        @test sol(0.001)≈[0.998687464, -1.312035941] atol=testTol
    end end
end end
