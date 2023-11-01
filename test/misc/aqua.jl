using Aqua, BoundaryValueDiffEq, Test

@testset "All Tests (except Ambiguities)" begin
    # Ambiguities are from downstream pacakges
    Aqua.test_all(BoundaryValueDiffEq; ambiguities = false)
end
