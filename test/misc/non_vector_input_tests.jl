@testitem "Non-Vector Inputs" begin
    using LinearAlgebra, NonlinearSolveFirstOrder, OrdinaryDiffEq

    for order in (2, 3, 4, 5, 6)
        s = Symbol("MIRK$(order)")
        @eval mirk_solver(::Val{$order}) = $(s)()
    end

    function f1!(du, u, p, t)
        du[1, 1] = u[1, 2]
        du[1, 2] = 0
    end

    function f1(u, p, t)
        return [u[1, 2] 0]
    end

    function boundary!(residual, u, p, t)
        residual[1, 1] = u(0.0)[1, 1] - 5
        residual[1, 2] = u(5.0)[1, 1]
    end

    function boundary_a!(resida, ua, p)
        resida[1, 1] = ua[1, 1] - 5
    end
    function boundary_b!(residb, ub, p)
        residb[1, 1] = ub[1, 1]
    end

    function boundary(u, p, t)
        return [u(0.0)[1, 1] - 5 u(5.0)[1, 1]]
    end

    boundary_a = (ua, p) -> [ua[1, 1] - 5]
    boundary_b = (ub, p) -> [ub[1, 1]]

    tspan = (0.0, 5.0)
    u0 = [5.0 -3.5]
    probs = [BVProblem(f1!, boundary!, u0, tspan; nlls = Val(false)),
        TwoPointBVProblem(f1!, (boundary_a!, boundary_b!), u0, tspan;
            bcresid_prototype = (Array{Float64}(undef, 1, 1), Array{Float64}(undef, 1, 1)),
            nlls = Val(false)),
        BVProblem(f1, boundary, u0, tspan; nlls = Val(false)),
        TwoPointBVProblem(f1, (boundary_a, boundary_b), u0, tspan; nlls = Val(false))]

    @testset "Affineness" begin
        @testset "MIRK$order" for order in (2, 3, 4, 5, 6)
            for prob in probs
                sol = solve(prob, mirk_solver(Val(order)); dt = 0.2)
                @test norm(boundary(sol, prob.p, nothing), Inf) < 0.01
            end
        end

        @testset "Single Shooting" begin
            for prob in probs
                sol = solve(prob, Shooting(Tsit5(), NewtonRaphson()))
                @test norm(boundary(sol, prob.p, nothing), Inf) < 0.01
            end
        end

        # FIXME: Add Multiple Shooting here once it supports non-vector inputs
        @testset "Multiple Shooting" begin
            for prob in probs
                @test_broken begin
                    sol = solve(prob, MultipleShooting(5, Tsit5()))
                    norm(boundary(sol, prob.p, nothing), Inf) < 0.01
                end
            end
        end
    end
end
