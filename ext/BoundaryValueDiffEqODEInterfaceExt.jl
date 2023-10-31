module BoundaryValueDiffEqODEInterfaceExt

using SciMLBase, BoundaryValueDiffEq, ODEInterface
import SciMLBase: __solve
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
## TODO: We can specify Drhs using forwarddiff if we want to
function __solve(prob::BVProblem, alg::BVPM2; dt = 0.0, reltol = 1e-3, kwargs...)
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
    bvpm2_init(initial_guess, no_odes, no_left_bc, mesh, u0, eltype(u0)[],
        alg.max_num_subintervals)

    bvp2m_f(t, u, du) = prob.f(du, u, prob.p, t)
    function bvp2m_bc(ya, yb, bca, bcb)
        prob.f.bc[1](bca, ya, prob.p)
        prob.f.bc[2](bcb, yb, prob.p)
        return nothing
    end

    opt = OptionsODE(OPT_RTOL => reltol, OPT_METHODCHOICE => alg.method_choice,
        OPT_DIAGNOSTICOUTPUT => alg.diagnostic_output,
        OPT_SINGULARTERM => alg.singular_term, OPT_ERRORCONTROL => alg.error_control)

    sol, retcode, stats = bvpm2_solve(initial_guess, bvp2m_f, bvp2m_bc, opt)
    retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure

    x_mesh = bvpm2_get_x(sol)
    sol_final = DiffEqBase.build_solution(prob, alg, x_mesh,
        eachcol(evalSolution(sol, x_mesh)); retcode, stats)

    bvpm2_destroy(initial_guess)
    bvpm2_destroy(sol)

    return sol_final
end

#-------
# BVPSOL
#-------
function __solve(prob::BVProblem, alg::BVPSOL; maxiters = 1000, reltol = 1e-3,
        dt = 0.0, verbose = true, kwargs...)
    _test_bvpm2_bvpsol_problem_criteria(prob, prob.problem_type, :BVPSOL)
    @assert isa(prob.p, SciMLBase.NullParameters) "BVPSOL only supports NullParameters!"
    @assert isa(prob.u0, AbstractVector{<:AbstractArray}) "BVPSOL requires a vector of initial guesses!"
    n, u0 = (length(prob.u0) - 1), reduce(hcat, prob.u0)
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))

    opt = OptionsODE(OPT_RTOL => reltol, OPT_MAXSTEPS => maxiters,
        OPT_BVPCLASS => alg.bvpclass, OPT_SOLMETHOD => alg.sol_method,
        OPT_RHS_CALLMODE => RHS_CALL_INSITU)

    f!(t, u, du) = prob.f(du, u, prob.p, t)
    function bc!(ya, yb, r)
        ra = first(prob.f.bcresid_prototype.x)
        rb = last(prob.f.bcresid_prototype.x)
        prob.f.bc[1](ra, ya, prob.p)
        prob.f.bc[2](rb, yb, prob.p)
        r[1:length(ra)] .= ra
        r[(length(ra) + 1):(length(ra) + length(rb))] .= rb
        return r
    end

    sol_t, sol_x, retcode, stats = bvpsol(f!, bc!, mesh, u0, alg.odesolver, opt)

    if verbose
        if retcode == -3
            @warn "Integrator failed to complete the trajectory"
        elseif retcode == -4
            @warn "Gauss Newton method failed to converge"
        elseif retcode == -5
            @warn "Given initial values inconsistent with separable linear bc"
        elseif retcode == -6
            @warn """Iterative refinement faild to converge for `sol_method=0`
            Termination since multiple shooting condition or
            condition of Jacobian is too bad for `sol_method=1`"""
        elseif retcode == -8
            @warn "Condensing algorithm for linear block system fails, try `sol_method=1`"
        elseif retcode == -9
            @warn "Sparse linear solver failed"
        elseif retcode == -10
            @warn "Real or integer work-space exhausted"
        elseif retcode == -11
            @warn "Rank reduction failed - resulting rank is zero"
        end
    end

    return DiffEqBase.build_solution(prob, alg, sol_t, eachcol(sol_x);
        retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure, stats)
end

end
