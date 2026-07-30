using SciMLTesting
using BoundaryValueDiffEq
using SciMLBase
using JET
using Test

const ROOT_FACADE_REEXPORTS = (
    :Ascher1, :Ascher2, :Ascher3, :Ascher4, :Ascher5, :Ascher6, :Ascher7,
    :BVPJacobianAlgorithm, :BVPVerbosity, :DEFAULT_VERBOSE,
    :LobattoIIIa2, :LobattoIIIa3, :LobattoIIIa4, :LobattoIIIa5,
    :LobattoIIIb2, :LobattoIIIb3, :LobattoIIIb4, :LobattoIIIb5,
    :LobattoIIIc2, :LobattoIIIc3, :LobattoIIIc4, :LobattoIIIc5,
    :MIRK2, :MIRK3, :MIRK4, :MIRK5, :MIRK6, :MIRKN4, :MIRKN6,
    :MultipleShooting,
    :RadauIIa1, :RadauIIa2, :RadauIIa3, :RadauIIa5, :RadauIIa7,
    :Shooting, :maxsol, :minsol,
)

@testset "Facade dependency bindings" begin
    @test !isdefined(BoundaryValueDiffEq, :BVProblem)
    @test !isdefined(BoundaryValueDiffEq, :AutoForwardDiff)
    @test !Base.ispublic(BoundaryValueDiffEq, :SciMLBase)
    @test !Base.ispublic(BoundaryValueDiffEq, :ADTypes)
end

run_qa(
    BoundaryValueDiffEq;
    aqua_kwargs = (;
        ambiguities = false,
        piracies = (; treat_as_own = [SciMLBase.BVProblem]),
    ),
    reexports_allow = ROOT_FACADE_REEXPORTS,
)
