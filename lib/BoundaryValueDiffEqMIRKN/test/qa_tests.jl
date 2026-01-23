@testitem "Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqMIRKN)
end

@testitem "JET Package Test" tags=[:qa] begin
    using JET

    JET.test_package(BoundaryValueDiffEqMIRKN, target_defined_modules = true)
end
