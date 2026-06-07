using BoundaryValueDiffEqMIRK
using Test

@testset "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqMIRK)
end
