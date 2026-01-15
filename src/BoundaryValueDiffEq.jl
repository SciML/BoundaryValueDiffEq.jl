module BoundaryValueDiffEq

using ADTypes
using BoundaryValueDiffEqAscher
using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqAlgorithm
using BoundaryValueDiffEqFIRK
using BoundaryValueDiffEqMIRK
using BoundaryValueDiffEqMIRKN
using BoundaryValueDiffEqShooting
using DiffEqBase: DiffEqBase, solve
using OrdinaryDiffEqTsit5: Tsit5
using Reexport: @reexport
using SciMLBase

@reexport using ADTypes, SciMLBase

function SciMLBase.__init(prob::BVProblem; kwargs...)
    return SciMLBase.__init(prob, Shooting(Tsit5()); kwargs...)
end

function SciMLBase.__solve(prob::BVProblem; kwargs...)
    return SciMLBase.__solve(prob, Shooting(Tsit5()); kwargs...)
end

include("extension_algs.jl")

export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6

export Shooting, MultipleShooting

export RadauIIa1, RadauIIa2, RadauIIa3, RadauIIa5, RadauIIa7
export LobattoIIIa2, LobattoIIIa3, LobattoIIIa4, LobattoIIIa5
export LobattoIIIb2, LobattoIIIb3, LobattoIIIb4, LobattoIIIb5
export LobattoIIIc2, LobattoIIIc3, LobattoIIIc4, LobattoIIIc5

export MIRKN4, MIRKN6

export Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7

export BVPM2, BVPSOL, COLNEW # From ODEInterface.jl

export BVPJacobianAlgorithm

export maxsol, minsol

end
