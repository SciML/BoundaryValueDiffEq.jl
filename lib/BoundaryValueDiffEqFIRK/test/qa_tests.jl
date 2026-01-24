@testitem "Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqFIRK)
end

@testitem "JET Package Test" tags=[:qa] begin
    import Pkg
    Pkg.add("JET")
    using JET

    JET.test_package(BoundaryValueDiffEqFIRK, target_defined_modules = true)
end
