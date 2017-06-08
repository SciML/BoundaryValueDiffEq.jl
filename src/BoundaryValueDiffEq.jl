__precompile__()

module BoundaryValueDiffEq

using Reexport
@reexport using DiffEqBase
  
using OrdinaryDiffEq
import DiffEqBase: solve
import NLsolve

abstract AbstractBVProblem{uType,tType,isinplace} <: DEProblem

type BVProblem{uType,tType,isinplace,F,bF} <: AbstractBVProblem{uType,tType,isinplace}
  f::F
  bc::bF
  u0::uType
  tspan::Tuple{tType,tType}
end

function BVProblem(f,bc,u0,tspan; iip = DiffEqBase.isinplace(f,3))
    BVProblem{typeof(u0),eltype(tspan),iip,typeof(f),typeof(bc)}(f,bc,u0,tspan)
end

include("algorithms.jl")
include("jacobian.jl")
include("solve.jl")

export BVProblem
export Shooting

end # module
