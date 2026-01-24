@testitem "Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(BoundaryValueDiffEqAscher)
end

@testitem "JET Package Test" tags=[:qa] begin
    import Pkg
    Pkg.add("JET")
    using JET

    JET.test_package(BoundaryValueDiffEqAscher, target_defined_modules = true)
end
