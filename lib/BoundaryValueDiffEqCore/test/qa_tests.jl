@testitem "Quality Assurance" tags = [:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqCore)
end
