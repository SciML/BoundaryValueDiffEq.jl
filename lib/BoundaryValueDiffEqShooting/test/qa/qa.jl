using BoundaryValueDiffEqShooting
using Aqua
using Test

@testset "Aqua" begin
    Aqua.test_all(BoundaryValueDiffEqShooting; persistent_tasks = false)
end
