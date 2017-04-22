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
  residual_prototype::initType
end

function BVProblem(f,domain,bc,init;residual_size=size(init,1))
  BVProblem{eltype(domain),typeof(bc),eltype(init),typeof(f)}(f,domain,bc,init,residual_prototype)
end

include("algorithms.jl")
include("jacobian.jl")
include("solve.jl")

export BVProblem
export Shooting

end # module
