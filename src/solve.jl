# The Solve Function
function solve(prob::BVProblem, alg::Shooting; kwargs...)
  bc = prob.bc
  u0 = deepcopy(prob.u0)
  # Form a root finding function.
  loss = function (minimizer,resid)
    uEltype = eltype(minimizer)
    tspan = (uEltype(prob.tspan[1]),uEltype(prob.tspan[2]))
    tmp_prob = ODEProblem(prob.f,minimizer,tspan)
    sol = solve(tmp_prob,alg.ode_alg;kwargs...)
    bc(resid,sol)
    nothing
  end
  opt = alg.nlsolve(loss, u0)
  sol_prob = ODEProblem(prob.f,opt,prob.tspan)
  solve(sol_prob, alg.ode_alg;kwargs...)
end

function solve(prob::BVProblem, alg::MIRK; kwargs...)
    n = Int(cld((prob.tspan[2]-prob.tspan[1]),alg.dt))
    x = collect(linspace(prob.tspan..., n+1))
    S = BVPSystem(prob.f, prob.bc, x, length(prob.u0), alg.order)
    S.y[1] = prob.u0
    # Upper-level iteration
    loss = function (z)
        z = nest_vector(z, S.M, S.N)
        copy!(S.y, z)
        Φ!(S)
        flatten_vector(S.residual)
    end
    opt = alg.nlsolve(loss, flatten_vector(S.y))
    retcode = opt[2] ? :Coverage : :Diverge
    y = nest_vector(opt[1], S.M, S.N)
    DiffEqBase.build_solution(prob, alg, x, y, retcode = retcode)
end
