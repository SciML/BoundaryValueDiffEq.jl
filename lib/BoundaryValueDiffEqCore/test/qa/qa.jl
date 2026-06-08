using BoundaryValueDiffEqCore
using Aqua
using Test

@testset "Aqua" begin
    Aqua.test_all(BoundaryValueDiffEqCore; piracies = false, ambiguities = false, stale_deps = false)
    Aqua.test_stale_deps(BoundaryValueDiffEqCore; ignore = [:TimerOutputs])
    Aqua.test_piracies(BoundaryValueDiffEqCore)
    Aqua.test_ambiguities(BoundaryValueDiffEqCore; recursive = false)
end
