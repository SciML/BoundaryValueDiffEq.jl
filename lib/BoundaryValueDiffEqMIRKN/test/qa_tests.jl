@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqMIRKN)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEqMIRKN, target_defined_modules = true)
end
