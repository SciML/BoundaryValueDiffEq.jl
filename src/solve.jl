# The Solve Function
function solve(prob::BVProblem, alg::Shooting; kwargs...)
  bc = prob.bc
  u0 = deepcopy(prob.init)
  # Convert a BVP Problem to a IVP problem.
  probIt = ODEProblem(prob.f, u0, prob.domain)
  # Form a root finding function.
  loss = function (minimizer,resid)
    uEltype = eltype(minimizer)
    tspan = (uEltype(prob.domain[1]),uEltype(prob.domain[2]))
    tmp_prob = ODEProblem(prob.f,minimizer,tspan)
    sol = solve(tmp_prob,alg.ode_alg;kwargs...)
    bc(resid,sol)
    nothing
  end
  diffsize = prob.residual_size - size(u0,1)
  if diffsize > 0
    vcat(u0, zeros(eltype(u0), diffsize))
  end
  opt = alg.nlsolve(loss, u0)[1:end-diffsize]
  sol_prob = ODEProblem(prob.f,opt,prob.domain)
  solve(sol_prob, alg.ode_alg;kwargs...)
end
