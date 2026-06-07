using BoundaryValueDiffEqShooting
using Test

@testset "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqShooting; persistent_tasks = false)
end
