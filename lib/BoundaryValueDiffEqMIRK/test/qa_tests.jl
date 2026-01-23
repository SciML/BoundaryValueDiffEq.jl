@testitem "Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqMIRK)
end

@testitem "JET Package Test" tags=[:qa] begin
    using JET

    JET.test_package(BoundaryValueDiffEqMIRK, target_defined_modules = true)
end
