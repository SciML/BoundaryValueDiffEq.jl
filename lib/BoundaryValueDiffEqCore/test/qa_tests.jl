using BoundaryValueDiffEqCore
using Test

@testset "Aqua" begin
    using Aqua, BoundaryValueDiffEqCore

    Aqua.test_all(BoundaryValueDiffEqCore; piracies = false, ambiguities = false, stale_deps = false)
    Aqua.test_stale_deps(BoundaryValueDiffEqCore; ignore = [:TimerOutputs])
    Aqua.test_piracies(BoundaryValueDiffEqCore)
    Aqua.test_ambiguities(BoundaryValueDiffEqCore; recursive = false)
end
