using BoundaryValueDiffEq
using Aqua
using SciMLBase
using Test

@testset "Aqua" begin
    Aqua.test_all(
        BoundaryValueDiffEq; ambiguities = false,
        piracies = (broken = false, treat_as_own = [SciMLBase.BVProblem])
    )
end
