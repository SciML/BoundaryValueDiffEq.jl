# The Solve Function
# The Solve Function
function solve(prob::BVProblem, alg::Shooting)
  bc = prob.bc
  u0 = bc[1]
  len = length(bc[1])
  # Convert a BVP Problem to a IVP problem.
  probIt = ODEProblem(prob.f, u0, prob.domain)
  # Form a root finding function.
  function loss(minimizer)
    probIt.u0 = minimizer
    sol = solve(probIt, alg.ode_alg)
    norm(sol[end]-bc[2])
  end
  opt = nlsolve(not_in_place(loss), u0)
  probIt.u0 = opt.zero
  solve(probIt, alg.ode_alg)
end
