@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEq; ambiguities = false)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(BoundaryValueDiffEq, target_defined_modules = true)
end
