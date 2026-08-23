using Test

include(joinpath(@__DIR__, "..", "..", "..", "test", "qa", "reexports.jl"))

module DocumentedFIRKWorkflow

    using BoundaryValueDiffEqFIRK

    const FIRK = BoundaryValueDiffEqFIRK

    function solve_documented_problems()
        function f!(du, u, p, t)
            du[1] = u[2]
            du[2] = 0
            return nothing
        end

        function bc!(residual, u, p, t)
            residual[1] = u(0.0)[1] - 1
            residual[2] = u(1.0)[1]
            return nothing
        end

        prob = BVProblem(f!, bc!, [1.0, -1.0], (0.0, 1.0); nlls = Val(false))
        sol = solve(prob, RadauIIa5(); dt = 0.2, abstol = 1.0e-8)

        function bca!(residual, u, p)
            residual[1] = u[1] - 1
            return nothing
        end

        function bcb!(residual, u, p)
            residual[1] = u[1]
            return nothing
        end

        two_point_prob = TwoPointBVProblem(
            f!, (bca!, bcb!), [1.0, -1.0], (0.0, 1.0);
            bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
        )
        two_point_sol = solve(two_point_prob, RadauIIa5(); dt = 0.2, abstol = 1.0e-8)

        return sol, two_point_sol
    end

end


@testset "Documented FIRK public facade" begin
    firk_algorithms = Set(
        [
            :RadauIIa1, :RadauIIa2, :RadauIIa3, :RadauIIa5, :RadauIIa7,
            :LobattoIIIa2, :LobattoIIIa3, :LobattoIIIa4, :LobattoIIIa5,
            :LobattoIIIb2, :LobattoIIIb3, :LobattoIIIb4, :LobattoIIIb5,
            :LobattoIIIc2, :LobattoIIIc3, :LobattoIIIc4, :LobattoIIIc5,
        ]
    )
    # The reexported surface is defined once in test/qa/reexports.jl and shared with
    # `reexports_allow` in test/qa/qa.jl, so this assertion cannot drift from it.
    documented_facade = Set(FIRK_REEXPORTS)
    actual_exports = Set(names(DocumentedFIRKWorkflow.FIRK))
    delete!(actual_exports, nameof(DocumentedFIRKWorkflow.FIRK))

    @test actual_exports == union(firk_algorithms, documented_facade)

    sol, two_point_sol = DocumentedFIRKWorkflow.solve_documented_problems()
    @test isapprox(sol(0.0)[1], 1.0; atol = 1.0e-6)
    @test isapprox(sol(1.0)[1], 0.0; atol = 1.0e-6)
    @test isapprox(two_point_sol(0.0)[1], 1.0; atol = 1.0e-6)
    @test isapprox(two_point_sol(1.0)[1], 0.0; atol = 1.0e-6)
end
