@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqFIRK)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEqFIRK, target_defined_modules = true)
end
