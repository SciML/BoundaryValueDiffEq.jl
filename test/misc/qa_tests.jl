@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEq; ambiguities = false)
    Aqua.test_ambiguities(BoundaryValueDiffEq; recursive = false)
end
