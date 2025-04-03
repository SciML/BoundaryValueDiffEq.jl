@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqAscher)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEqAscher, target_defined_modules = true)
end
