@testitem "Quality Assurance" begin
    using Aqua

    # Report unrealted deprecated warnings
    # @test_broken Aqua.test_all(BoundaryValueDiffEq; ambiguities = false)
    # @test_broken Aqua.test_ambiguities(BoundaryValueDiffEq; recursive = false)
end
