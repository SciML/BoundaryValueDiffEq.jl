@testitem "Quality Assurance" tags = [:qa] begin
    using Aqua

    Aqua.test_all(
        BoundaryValueDiffEq; ambiguities = false,
        piracies = (broken = false, treat_as_own = [SciMLBase.BVProblem])
    )
end

@testitem "JET Package Test" tags = [:qa] begin
    import Pkg
    Pkg.add("JET")
    using JET

    JET.test_package(BoundaryValueDiffEq, target_defined_modules = true)
end
