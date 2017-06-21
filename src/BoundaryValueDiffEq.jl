# __precompile__()

module BoundaryValueDiffEq

using Reexport
@reexport using DiffEqBase
  
using OrdinaryDiffEq
import DiffEqBase: solve
import NLsolve

abstract type AbstractBVProblem{uType,tType,isinplace} <: AbstractODEProblem{uType,tType,isinplace} end

type BVProblem{uType,tType,isinplace,F,bF} <: AbstractBVProblem{uType,tType,isinplace}
  f::F
  bc::bF
  u0::uType
  tspan::Tuple{tType,tType}
end

function BVProblem(f,bc,u0,tspan; iip = DiffEqBase.isinplace(f,3))
    BVProblem{typeof(u0),eltype(tspan),iip,typeof(f),typeof(bc)}(f,bc,u0,tspan)
end

immutable MIRKTableau{T, U<:AbstractArray}
    c::Vector{T}
    v::Vector{T}
    b::Vector{T}
    x::Matrix{T}
    K::Vector{U} # Cache
end

# ODE BVP problem system
immutable BVPSystem{T,U<:AbstractArray}
    order::Int                  # The order of MIRK method
    M::Int                      # Number of equations in the ODE system
    N::Int                      # Number of nodes in the mesh
    fun!                        # M -> M
    bc!                         # 2 -> 2
    x::Vector{T}                # N
    y::Vector{U}                # N{M}
    f::Vector{U}                # N{M}
    residual::Vector{U}         # N{M}
end

include("vector_auxiliary.jl")
include("algorithms.jl")
# include("jacobian.jl")
include("mirk_tableaus.jl")
include("collocation.jl")
include("solve.jl")

export BVProblem
export Shooting
export MIRK

end # module
