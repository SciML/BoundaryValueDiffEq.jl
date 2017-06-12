# __precompile__()

module BoundaryValueDiffEq

using Reexport
@reexport using DiffEqBase
  
using OrdinaryDiffEq
import DiffEqBase: solve
import NLsolve

abstract AbstractBVProblem{uType,tType,isinplace} <: AbstractODEProblem{uType,tType,isinplace}

type BVProblem{uType,tType,isinplace,F,bF} <: AbstractBVProblem{uType,tType,isinplace}
  f::F
  bc::bF
  u0::uType
  tspan::Tuple{tType,tType}
end

function BVProblem(f,bc,u0,tspan; iip = DiffEqBase.isinplace(f,3))
    BVProblem{typeof(u0),eltype(tspan),iip,typeof(f),typeof(bc)}(f,bc,u0,tspan)
end

immutable MIRKTableau{T}
    c::Vector{T}
    v::Vector{T}
    b::Vector{T}
    x::Matrix{T}
    K::Matrix{T} # Cache
end

# ODE BVP problem system
immutable BVPSystem{T}  # Order of the system
    order::Int          # The order of MIRK method
    M::Int              # Number of equations in the ODE system
    N::Int              # Number of nodes in the mesh
    fun!::Function      # M -> M
    bc!::Function       # 2 -> 2
    x::Vector{T}        # N
    y::Vector{Vector{T}}        # M*N
    f::Vector{Vector{T}}        # M*N
    residual::Vector{Vector{T}} # M*N
end

include("algorithms.jl")
# include("jacobian.jl")
include("mirk_tableaus.jl")
include("collocation.jl")
include("solve.jl")

export BVProblem
export Shooting
export MIRK

end # module
