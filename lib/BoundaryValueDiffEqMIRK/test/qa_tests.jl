@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqMIRK)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEqMIRK, target_defined_modules = true)
end
