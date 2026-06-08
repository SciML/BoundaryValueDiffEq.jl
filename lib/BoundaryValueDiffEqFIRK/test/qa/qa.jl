using BoundaryValueDiffEqFIRK
using Aqua
using Test

@testset "Aqua" begin
    Aqua.test_all(BoundaryValueDiffEqFIRK)
end
