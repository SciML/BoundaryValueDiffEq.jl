using BoundaryValueDiffEqMIRK
using Aqua
using Test

@testset "Aqua" begin
    Aqua.test_all(BoundaryValueDiffEqMIRK)
end
