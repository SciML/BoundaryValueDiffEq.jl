module BoundaryValueDiffEqODEInterfaceExt

using SciMLBase, BoundaryValueDiffEq, ODEInterface
import ODEInterface: OptionsODE, OPT_ATOL, OPT_RTOL, OPT_METHODCHOICE, OPT_DIAGNOSTICOUTPUT,
    OPT_ERRORCONTROL, OPT_SINGULARTERM, OPT_MAXSTEPS, OPT_BVPCLASS, OPT_SOLMETHOD,
    OPT_RHS_CALLMODE, RHS_CALL_INSITU, evalSolution
import ODEInterface: Bvpm2, bvpm2_init, bvpm2_solve, bvpm2_destroy, bvpm2_get_x
import ODEInterface: bvpsol

function _test_bvpm2_bvpsol_problem_criteria(_, ::SciMLBase.StandardBVProblem, alg::Symbol)
    throw(ArgumentError("$(alg) does not support standard BVProblem. Only TwoPointBVProblem is supported."))
end
function _test_bvpm2_bvpsol_problem_criteria(prob, ::TwoPointBVProblem, alg::Symbol)
    @assert isinplace(prob) "$(alg) only supports inplace TwoPointBVProblem!"
end

#------
# BVPM2
#------
_no_param(::SciMLBase.NullParameters) = Float64[]
_no_param(p) = p

bvpm2_bc(bc, ya, yb, bca, bcb) = bc((bca, bcb), (ya, yb), SciMLBase.NullParameters())
bvpm2_bc(bc, ya, yb, p, bca, bcb) = bc((bca, bcb), (ya, yb), p)

bvp2m_f(f, t, u, du) = f(du, u, SciMLBase.NullParameters(), t)
bvp2m_f(f, t, u, p, du) = f(du, u, p, t)

## TODO: We can specify Drhs using forwarddiff if we want to
function SciMLBase.__solve(prob::BVProblem, alg::BVPM2; dt = 0.0, reltol = 1e-3, kwargs...)
    _test_bvpm2_bvpsol_problem_criteria(prob, prob.problem_type, :BVPM2)

    has_initial_guess = prob.u0 isa AbstractVector{<:AbstractArray}
    no_odes, n, u0 = if has_initial_guess
        length(first(prob.u0)), (length(prob.u0) - 1), reduce(hcat, prob.u0)
    else
        dt ≤ 0 && throw(ArgumentError("dt must be positive"))
        length(prob.u0), Int(cld((prob.tspan[2] - prob.tspan[1]), dt)), prob.u0
    end

    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))

    no_left_bc = length(first(prob.f.bcresid_prototype.x))

    initial_guess = Bvpm2()
    bvpm2_init(initial_guess, no_odes, no_left_bc, mesh, u0, _no_param(prob.p),
        alg.max_num_subintervals)

    rhs = (args...) -> bvp2m_f(prob.f, args...)
    bc = (args...) -> bvpm2_bc(prob.bc, args...)

    opt = OptionsODE(OPT_RTOL => reltol, OPT_METHODCHOICE => alg.method_choice,
        OPT_DIAGNOSTICOUTPUT => alg.diagnostic_output,
        OPT_SINGULARTERM => alg.singular_term, OPT_ERRORCONTROL => alg.error_control)

    sol, retcode, stats = bvpm2_solve(initial_guess, rhs, bc, opt)
    retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure
    bvpm2_destroy(initial_guess)

    x_mesh = bvpm2_get_x(sol)
    return DiffEqBase.build_solution(prob, alg, x_mesh, eachcol(evalSolution(sol, x_mesh));
        retcode, stats)
end

#-------
# BVPSOL
#-------
bvpsol_f(f, t, u, du) = f(du, u, SciMLBase.NullParameters(), t)
function bvpsol_bc(bc, ra, rb, ya, yb, r)
    bc((view(r, 1:(length(ra))), view(r, (length(ra) + 1):(length(ra) + length(rb)))),
        (ya, yb), SciMLBase.NullParameters())
end

function SciMLBase.__solve(prob::BVProblem, alg::BVPSOL; maxiters = 1000, reltol = 1e-3,
    dt = 0.0, kwargs...)
    _test_bvpm2_bvpsol_problem_criteria(prob, prob.problem_type, :BVPSOL)
    @assert isa(prob.p, SciMLBase.NullParameters) "BVPSOL only supports NullParameters!"
    @assert isa(prob.u0, AbstractVector{<:AbstractArray}) "BVPSOL requires a vector of initial guesses!"
    n, u0 = (length(prob.u0) - 1), reduce(hcat, prob.u0)
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))

    opt = OptionsODE(OPT_RTOL => reltol, OPT_MAXSTEPS => maxiters,
        OPT_BVPCLASS => alg.bvpclass, OPT_SOLMETHOD => alg.sol_method,
        OPT_RHS_CALLMODE => RHS_CALL_INSITU)

    f! = (args...) -> bvpsol_f(prob.f, args...)
    bc! = (args...) -> bvpsol_bc(prob.bc, first(prob.f.bcresid_prototype.x),
        last(prob.f.bcresid_prototype.x), args...)

    sol_t, sol_x, retcode, stats = bvpsol(f!, bc!, mesh, u0, alg.odesolver, opt)

    return DiffEqBase.build_solution(prob, alg, sol_t, eachcol(sol_x);
        retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure, stats)
end

end
