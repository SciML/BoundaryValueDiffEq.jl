module BoundaryValueDiffEq

using DiffEqBase, OrdinaryDiffEq
import DiffEqBase: solve
import NLsolve

abstract AbstractBVProblem{dType,bF,isinplace,F} <: DEProblem

type BVProblem{dType,bF,initType,F} <: AbstractBVProblem{dType,bF,F}
  f::F
  domain::dType
  bc::bF
  init::initType
end

function BVProblem(f,domain,bc,init)
  BVProblem{eltype(domain),typeof(bc),eltype(init),typeof(f)}(f,domain,bc,init)
end

include("algorithms.jl")
include("solve.jl")

export BVProblem
export Shooting

end # module
