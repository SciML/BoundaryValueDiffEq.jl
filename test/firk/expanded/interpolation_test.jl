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
nested = false

@testset "Radau interpolations" begin
    for stage in (2, 3, 5, 7)
        s = Symbol("RadauIIa$(stage)")
        @eval radau_solver(::Val{$stage}) = $(s)(NewtonRaphson(),BVPJacobianAlgorithm(); nested)
    end
    @testset "Interpolation" begin @testset "RadauIIa$stage" for stage in (2, 3, 5, 7)
        @time sol = solve(prob_bvp_linear, radau_solver(Val(stage)); dt = 0.001)
        if stage == 2
            @test sol(0.001)≈[0.998687464, -1.312035941] atol=testTol 
        else
            @test sol(0.001)≈[0.998687464, -1.312035941] atol=testTol
        end
    end end
end

@testset "LobattoIII interpolations" begin for lobatto in ("a", "b", "c")
    for stage in (3, 4, 5)
        s = Symbol("LobattoIII$(lobatto)$(stage)")
        @eval lobatto_solver(::Val{$stage}) = $(s)(NewtonRaphson(),BVPJacobianAlgorithm(); nested)
    end

    @testset "Interpolation" begin @testset "LobattoIII$(lobatto)$stage" for stage in (3,
                                                                                       4, 5)
        @time sol = solve(prob_bvp_linear, lobatto_solver(Val(stage)); dt = 0.001)
        @test sol(0.001)≈[0.998687464, -1.312035941] atol=testTol
    end end
end end
