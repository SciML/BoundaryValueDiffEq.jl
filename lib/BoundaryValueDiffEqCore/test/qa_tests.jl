@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqCore)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEqCore, target_defined_modules = true)
end
