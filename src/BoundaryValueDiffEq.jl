module BoundaryValueDiffEq

# using DiffEqBase, OrdinaryDiffEq
# import DiffEqBase: solve
using DifferentialEquations
using Optim

abstract AbstractBVProblem{dType,bType,isinplace,F} <: DEProblem

type BVProblem{dType,bType,initType,F} <: AbstractBVProblem{dType,bType,F}
  f::F
  domin::dType
  bc::bType
  init::initType
end

function BVProblem(f,domin,bc,init=nothing)
  BVProblem{eltype(domin),eltype(bc),eltype(init),typeof(f)}(f,domin,bc,init)
end

function solve(prob::BVProblem; OptSolver=LBFGS())#, ODESolver...)
  bc = prob.bc
  u0 = bc[1]
  len = length(bc[1])
  probIt = ODEProblem(prob.f, u0, prob.domin)
  function loss(minimizer)
    probIt.u0 = minimizer
    sol = DifferentialEquations.solve(probIt)#, ODESolver...)
    norm(sol[end]-bc[2])
  end
  opt = optimize(loss, u0, OptSolver)
  probIt.u0 = opt.minimizer
  @show opt.minimum
  DifferentialEquations.solve(probIt)
end

end # module
