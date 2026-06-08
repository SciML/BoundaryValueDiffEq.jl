using BoundaryValueDiffEq
using JET
using Test

@testset "JET" begin
    rep = JET.report_package(
        BoundaryValueDiffEq;
        target_modules = (BoundaryValueDiffEq,)
    )
    @test length(JET.get_reports(rep)) == 0
end
