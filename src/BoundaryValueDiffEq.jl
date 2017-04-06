module BoundaryValueDiffEq

using DiffEqBase, OrdinaryDiffEq
import DiffEqBase: solve
# using DifferentialEquations
using NLsolve

abstract AbstractBVProblem{dType,bType,isinplace,F} <: DEProblem

type BVProblem{dType,bType,initType,F} <: AbstractBVProblem{dType,bType,F}
  f::F
  domain::dType
  bc::bType
  init::initType
end

function BVProblem(f,domain,bc,init=nothing)
  BVProblem{eltype(domain),eltype(bc),eltype(init),typeof(f)}(f,domain,bc,init)
end

include("algorithms.jl")
include("solve.jl")

end # module
