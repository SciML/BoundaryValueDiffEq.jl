@testitem "Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqCore)
end

@testitem "JET Package Test" tags=[:qa] begin
    import Pkg
    Pkg.add("JET")
    using JET

    JET.test_package(BoundaryValueDiffEqCore, target_defined_modules = true)
end
