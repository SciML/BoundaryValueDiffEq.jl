@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(
        BoundaryValueDiffEq; ambiguities = false,
        piracies = (broken = false, treat_as_own = [SciMLBase.BVProblem])
    )
end
