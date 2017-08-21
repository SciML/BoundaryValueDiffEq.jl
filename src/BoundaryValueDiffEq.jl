# __precompile__()

module BoundaryValueDiffEq

using Reexport
@reexport using DiffEqBase

using OrdinaryDiffEq
import DiffEqBase: solve
import NLsolve, ForwardDiff, BandedMatrices, Sundials, DiffEqDiffTools

abstract type AbstractBVProblem{uType,tType,isinplace} <: AbstractODEProblem{uType,tType,isinplace} end

struct BVProblem{uType,tType,isinplace,F,bF} <: AbstractBVProblem{uType,tType,isinplace}
    f::F
    bc::bF
    u0::uType
    tspan::Tuple{tType,tType}
end

function BVProblem(f,bc,u0,tspan; iip = DiffEqBase.isinplace(f,3))
    BVProblem{typeof(u0),eltype(tspan),iip,typeof(f),typeof(bc)}(f,bc,u0,tspan)
end

struct TwoPointBVProblem{uType,tType,isinplace,F,bF} <: AbstractBVProblem{uType,tType,isinplace}
    f::F
    bc::bF
    u0::uType
    tspan::Tuple{tType,tType}
end

function TwoPointBVProblem(f,bc,u0,tspan; iip = DiffEqBase.isinplace(f,3))
    TwoPointBVProblem{typeof(u0),eltype(tspan),iip,typeof(f),typeof(bc)}(f,bc,u0,tspan)
end

struct MIRKTableau{T}
    c::Vector{T}
    v::Vector{T}
    b::Vector{T}
    x::Matrix{T}
end

# ODE BVP problem system
mutable struct BVPSystem{T,U<:AbstractArray}
    order::Int                  # The order of MIRK method
    M::Int                      # Number of equations in the ODE system
    N::Int                      # Number of nodes in the mesh
    fun!                        # M -> M
    bc!                         # 2 -> 2
    x::Vector{T}                # N
    y::Vector{U}                # N{M}
    f::Vector{U}                # N{M}
    residual::Vector{U}         # N{M}
    tmp::Vector{T}
end

include("vector_auxiliary.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("mirk_tableaus.jl")
include("cache.jl")
include("collocation.jl")
include("jacobian.jl")
include("solve.jl")

export BVProblem, TwoPointBVProblem
export Shooting
export MIRK4, GeneralMIRK4

end # module
