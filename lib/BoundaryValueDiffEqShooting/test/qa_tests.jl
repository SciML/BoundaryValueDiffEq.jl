@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqShooting)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEqShooting, target_defined_modules = true)
end
