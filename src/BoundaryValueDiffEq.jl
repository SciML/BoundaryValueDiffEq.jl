module BoundaryValueDiffEq

using BandedMatrices, LinearAlgebra, Reexport, Setfield, SparseArrays
@reexport using DiffEqBase, NonlinearSolve

import DiffEqBase: solve
import ForwardDiff, BandedMatrices, FiniteDiff

import TruncatedStacktraces: @truncate_stacktrace

struct MIRKTableau{sType, cType, vType, bType, xType}
    """Discrete stages of MIRK formula"""
    s::sType
    c::cType
    v::vType
    b::bType
    x::xType

    function MIRKTableau(s, c, v, b, x)
        @assert eltype(c) == eltype(v) == eltype(b) == eltype(x)
        return new{typeof(s),
            typeof(c),
            typeof(v),
            typeof(b),
            typeof(x)}(s, c, v, b, x)
    end
end

@truncate_stacktrace MIRKTableau 1

struct MIRKInterpTableau{s, c, v, x, τ}
    s_star::s
    c_star::c
    v_star::v
    x_star::x
    τ_star::τ

    function MIRKInterpTableau(s_star, c_star, v_star, x_star, τ_star)
        @assert eltype(c_star) == eltype(v_star) == eltype(x_star)
        return new{typeof(s_star),
            typeof(c_star),
            typeof(v_star),
            typeof(x_star),
            typeof(τ_star)}(s_star,
            c_star,
            v_star,
            x_star,
            τ_star)
    end
end

@truncate_stacktrace MIRKInterpTableau 1

# ODE BVP problem system
struct BVPSystem{T, U <: AbstractArray, P, F, B, S}
    order::Int                  # The order of MIRK method
    M::Int                      # Number of equations in the ODE system
    N::Int                      # Number of nodes in the mesh
    fun!::F                     # M -> M
    bc!::B                      # 2 -> 2
    p::P
    s::S
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
include("adaptivity.jl")

export Shooting
export GeneralMIRK3, GeneralMIRK4, GeneralMIRK5, GeneralMIRK6
export MIRK3, MIRK4, MIRK5, MIRK6

end
