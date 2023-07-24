module BoundaryValueDiffEq

using BandedMatrices, LinearAlgebra, Reexport, Setfield, SparseArrays
@reexport using DiffEqBase, NonlinearSolve

import DiffEqBase: solve
import ForwardDiff, BandedMatrices, FiniteDiff

import TruncatedStacktraces: @truncate_stacktrace

struct MIRKTableau{T, cType, vType, bType, xType, sType, starType, tauType}
    c::cType
    v::vType
    b::bType
    x::xType
    """Discrete stages of MIRK formula"""
    s::sType
    """Number of stages to form the interpolant"""
    s_star::starType
    """Defect sampling point"""
    tau::tauType
end

function MIRKTableau(c, v, b, x, s, s_star, tau)
    @assert eltype(c) == eltype(v) == eltype(b) == eltype(x) == eltype(tau)
    return MIRKTableau{
        eltype(c),
        typeof(c),
        typeof(v),
        typeof(b),
        typeof(x),
        typeof(s),
        typeof(s_star),
        typeof(tau),
    }(c,
        v,
        b,
        x,
        s,
        s_star,
        tau)
end

@truncate_stacktrace MIRKTableau 1

# ODE BVP problem system
mutable struct BVPSystem{T, U <: AbstractArray, P, F, B, S}
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
export GeneralMIRK4, GeneralMIRK5, GeneralMIRK6
export MIRK4, MIRK5, MIRK6

end
