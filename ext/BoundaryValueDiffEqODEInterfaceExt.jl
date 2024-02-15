module BoundaryValueDiffEqODEInterfaceExt

using SciMLBase, BoundaryValueDiffEq, ODEInterface
import SciMLBase: __solve
import ODEInterface: OptionsODE, OPT_ATOL, OPT_RTOL, OPT_METHODCHOICE, OPT_DIAGNOSTICOUTPUT,
                     OPT_ERRORCONTROL, OPT_SINGULARTERM, OPT_MAXSTEPS, OPT_BVPCLASS,
                     OPT_SOLMETHOD,
                     OPT_RHS_CALLMODE, OPT_COLLOCATIONPTS, OPT_MAXSUBINTERVALS,
                     RHS_CALL_INSITU, evalSolution
import ODEInterface: Bvpm2, bvpm2_init, bvpm2_solve, bvpm2_destroy, bvpm2_get_x
import ODEInterface: bvpsol
import ODEInterface: colnew

import ForwardDiff

function _test_bvpm2_bvpsol_colnew_problem_criteria(
        _, ::SciMLBase.StandardBVProblem, alg::Symbol)
    throw(ArgumentError("$(alg) does not support standard BVProblem. Only TwoPointBVProblem is supported."))
end
function _test_bvpm2_bvpsol_colnew_problem_criteria(prob, ::TwoPointBVProblem, alg::Symbol)
    @assert isinplace(prob) "$(alg) only supports inplace TwoPointBVProblem!"
end

#------
# BVPM2
#------
## TODO: We can specify Drhs using forwarddiff if we want to
function SciMLBase.__solve(prob::BVProblem, alg::BVPM2; dt = 0.0, reltol = 1e-3, kwargs...)
    _test_bvpm2_bvpsol_colnew_problem_criteria(prob, prob.problem_type, :BVPM2)

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
    destats = SciMLBase.DEStats(
        stats["no_rhs_calls"], 0, 0, 0, stats["no_jac_calls"], 0, 0, 0, 0, 0, 0)

    x_mesh = bvpm2_get_x(sol)
    evalsol = evalSolution(sol, x_mesh)
    sol_final = DiffEqBase.build_solution(prob, alg, x_mesh,
        collect(Vector{eltype(evalsol)}, eachcol(evalsol)); retcode, stats = destats)

    bvpm2_destroy(initial_guess)
    bvpm2_destroy(sol)

    return sol_final
end

#-------
# BVPSOL
#-------
function SciMLBase.__solve(prob::BVProblem, alg::BVPSOL; maxiters = 1000, reltol = 1e-3,
        dt = 0.0, verbose = true, kwargs...)
    _test_bvpm2_bvpsol_colnew_problem_criteria(prob, prob.problem_type, :BVPSOL)
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

    return DiffEqBase.build_solution(prob, alg, sol_t,
        collect(Vector{eltype(sol_x)}, eachcol(sol_x));
        retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure, stats)
end

#-------
# COLNEW
#-------
function SciMLBase.__solve(prob::BVProblem, alg::COLNEW; maxiters = 1000,
        reltol = 1e-4, dt = 0.0, verbose = true, kwargs...)
    _test_bvpm2_bvpsol_colnew_problem_criteria(prob, prob.problem_type, :COLNEW)
    has_initial_guess = prob.u0 isa AbstractVector{<:AbstractArray}
    dt ≤ 0 && throw(ArgumentError("dt must be positive"))
    no_odes, n, u0 = if has_initial_guess
        length(first(prob.u0)), (length(prob.u0) - 1), reduce(hcat, prob.u0)
    else
        length(prob.u0), Int(cld((prob.tspan[2] - prob.tspan[1]), dt)), prob.u0
    end
    T = eltype(u0)
    mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
    opt = OptionsODE(
        OPT_BVPCLASS => alg.bvpclass, OPT_COLLOCATIONPTS => alg.collocationpts,
        OPT_MAXSTEPS => maxiters, OPT_DIAGNOSTICOUTPUT => alg.diagnostic_output,
        OPT_MAXSUBINTERVALS => alg.max_num_subintervals, OPT_RTOL => reltol)
    orders = ones(Int, no_odes)
    _tspan = [prob.tspan[1], prob.tspan[2]]
    iip = SciMLBase.isinplace(prob)

    rhs(t, u, du) =
        if iip
            prob.f(du, u, prob.p, t)
        else
            (du .= prob.f(u, prob.p, t))
        end

    if prob.f.jac === nothing
        if iip
            jac = function (df, u, p, t)
                _du = similar(u)
                prob.f(_du, u, p, t)
                _f = (du, u) -> prob.f(du, u, p, t)
                ForwardDiff.jacobian!(df, _f, _du, u)
            end
        else
            jac = function (df, u, p, t)
                _du = prob.f(u, p, t)
                _f = (du, u) -> (du .= prob.f(u, p, t))
                ForwardDiff.jacobian!(df, _f, _du, u)
            end
        end
    else
        jac = prob.f.jac
    end
    Drhs(t, u, df) = jac(df, u, prob.p, t)

    #TODO: Fix bc and bcjac for multi-points BVP

    n_bc_a = length(first(prob.f.bcresid_prototype.x))
    n_bc_b = length(last(prob.f.bcresid_prototype.x))
    zeta = vcat(fill(first(prob.tspan), n_bc_a), fill(last(prob.tspan), n_bc_b))
    bc = function (i, z, resid)
        tmpa = copy(z)
        tmpb = copy(z)
        tmp_resid_a = zeros(T, n_bc_a)
        tmp_resid_b = zeros(T, n_bc_b)
        prob.f.bc[1](tmp_resid_a, tmpa, prob.p)
        prob.f.bc[2](tmp_resid_b, tmpb, prob.p)

        for j in 1:n_bc_a
            if i == j
                resid[1] = tmp_resid_a[j]
            end
        end
        for j in 1:n_bc_b
            if i == (j + n_bc_a)
                resid[1] = tmp_resid_b[j]
            end
        end
    end

    Dbc = function (i, z, dbc)
        for j in 1:n_bc_a
            if i == j
                dbc[i] = 1.0
            end
        end
        for j in 1:n_bc_b
            if i == (j + n_bc_a)
                dbc[i] = 1.0
            end
        end
    end

    sol, retcode, stats = colnew(_tspan, orders, zeta, rhs, Drhs, bc, Dbc, nothing, opt)

    if verbose
        if retcode == 0
            @warn "Collocation matrix is singular"
        elseif retcode == -1
            @warn "The expected no. of subintervals exceeds storage(try to increase `OPT_MAXSUBINTERVALS`)"
        elseif retcode == -2
            @warn "The nonlinear iteration has not converged"
        elseif retcode == -3
            @warn "There is an input data error"
        end
    end

    evalsol = evalSolution(sol, mesh)
    destats = SciMLBase.DEStats(
        stats["no_rhs_calls"], 0, 0, 0, stats["no_jac_calls"], 0, 0, 0, 0, 0, 0)

    return DiffEqBase.build_solution(prob, alg, mesh,
        collect(Vector{eltype(evalsol)}, eachrow(evalsol));
        retcode = retcode > 0 ? ReturnCode.Success : ReturnCode.Failure,
        stats = destats)
end

end
