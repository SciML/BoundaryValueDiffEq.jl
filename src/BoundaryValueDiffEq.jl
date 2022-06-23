module BoundaryValueDiffEq

using Reexport, LinearAlgebra, SparseArrays
@reexport using DiffEqBase

import DiffEqBase: solve
using NLsolve: NLsolve
using ForwardDiff: ForwardDiff
using BandedMatrices: BandedMatrices
using FiniteDiff: FiniteDiff

struct MIRKTableau{T}
    c::Vector{T}
    v::Vector{T}
    b::Vector{T}
    x::Matrix{T}
end

# ODE BVP problem system
mutable struct BVPSystem{T,U<:AbstractArray,P}
    order::Int                  # The order of MIRK method
    M::Int                      # Number of equations in the ODE system
    N::Int                      # Number of nodes in the mesh
    fun!::Any                        # M -> M
    bc!::Any                         # 2 -> 2
    p::P
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
export MIRK4, GeneralMIRK4

end # module
