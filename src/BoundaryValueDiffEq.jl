module BoundaryValueDiffEq

using BandedMatrices, LinearAlgebra, Reexport, Setfield, SparseArrays
@reexport using DiffEqBase, NonlinearSolve

import DiffEqBase: solve
import ForwardDiff, BandedMatrices, FiniteDiff

import TruncatedStacktraces: @truncate_stacktrace

struct MIRKTableau{T, cType, vType, bType, xType}
    c::cType
    v::vType
    b::bType
    x::xType
end

function MIRKTableau(c, v, b, x)
    @assert eltype(c) == eltype(v) == eltype(b) == eltype(x)
    return MIRKTableau{eltype(c), typeof(c), typeof(v), typeof(b), typeof(x)}(c, v, b, x)
end

@truncate_stacktrace MIRKTableau 1

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

export Shooting
export GeneralMIRK4, GeneralMIRK6
export MIRK4, MIRK6

end
