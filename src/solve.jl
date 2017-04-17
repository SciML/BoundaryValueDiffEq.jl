# The Solve Function
function solve(prob::BVProblem, alg::Shooting; kwargs...)
  bc = prob.bc
  u0 = deepcopy(prob.init)
  # Convert a BVP Problem to a IVP problem.
  probIt = ODEProblem(prob.f, u0, prob.domain)
  # Form a root finding function.
  function loss(minimizer,boundary)
    copy!(probIt.u0, minimizer)
    sol = solve(probIt, alg.ode_alg)
    bc(boundary,sol)
    nothing
  end
  opt = nlsolve(loss, u0)
  probIt.u0 = opt.zero
  solve(probIt, alg.ode_alg)
end
