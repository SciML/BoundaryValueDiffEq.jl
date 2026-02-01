@testitem "Quality Assurance" tags = [:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqShooting; persistent_tasks = false)
end
