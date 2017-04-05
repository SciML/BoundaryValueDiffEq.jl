module BoundaryValueDiffEq

using DiffEqBase, OrdinaryDiffEq
import DiffEqBase: solve
# using DifferentialEquations
using Optim

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

function solve(prob::BVProblem; OptSolver=LBFGS(), ODESolver=Tsit5())
  bc = prob.bc
  u0 = bc[1]
  len = length(bc[1])
  # Convert a BVP Problem to a IVP problem.
  probIt = ODEProblem(prob.f, u0, prob.domain)
  # Form a root finding function.
  function loss(minimizer)
    probIt.u0 = minimizer
    sol = solve(probIt, ODESolver)
    norm(sol[end]-bc[2])
  end
  opt = optimize(loss, u0, OptSolver)
  probIt.u0 = opt.minimizer
  opt.minimum
  solve(probIt, ODESolver)
end

end # module
