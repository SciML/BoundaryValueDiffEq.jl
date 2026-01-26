using BoundaryValueDiffEq, Aqua, JET, Test, SciMLBase

@testset "Quality Assurance" begin
    @testset "Aqua" begin
        Aqua.test_all(
            BoundaryValueDiffEq; ambiguities = false,
            piracies = (broken = false, treat_as_own = [SciMLBase.BVProblem])
        )
    end

    @testset "JET" begin
        rep = JET.report_package(
            BoundaryValueDiffEq;
            target_modules = (BoundaryValueDiffEq,)
        )
        @test length(JET.get_reports(rep)) == 0
    end
end
