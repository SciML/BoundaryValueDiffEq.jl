module BoundaryValueDiffEqODEInterfaceExt

using SciMLBase, BoundaryValueDiffEq, ODEInterface, RecursiveArrayTools
import BoundaryValueDiffEq: __extract_u0, __flatten_initial_guess,
    __extract_mesh, __initial_guess_length, __initial_guess, __has_initial_guess
import SciMLBase: __solve
import ODEInterface: OptionsODE, OPT_ATOL, OPT_RTOL, OPT_METHODCHOICE, OPT_DIAGNOSTICOUTPUT,
    OPT_ERRORCONTROL, OPT_SINGULARTERM, OPT_MAXSTEPS, OPT_BVPCLASS, OPT_SOLMETHOD,
    OPT_RHS_CALLMODE, RHS_CALL_INSITU, evalSolution
import ODEInterface: Bvpm2, bvpm2_init, bvpm2_solve, bvpm2_destroy, bvpm2_get_x
import ODEInterface: bvpsol
import FastClosures: @closure

#------
# BVPM2
#------
function __solve(prob::BVProblem, alg::BVPM2; dt = 0.0, reltol = 1e-3, kwargs...)
    if !(prob.problem_type isa TwoPointBVProblem)
        throw(ArgumentError("`BVPM2` only supports `TwoPointBVProblem!`"))
    end

    t₀, t₁ = prob.tspan
    u0_ = __extract_u0(prob.u0, prob.p, t₀)
    u0_size = size(u0_)
    n = __initial_guess_length(prob.u0)

    n == -1 && dt ≤ 0 && throw(ArgumentError("`dt` must be positive."))

    mesh = __extract_mesh(prob.u0, t₀, t₁, ifelse(n == -1, dt, n))
    n = length(mesh) - 1
    no_odes = length(u0_)

    if prob.f.bcresid_prototype !== nothing
        left_bc, right_bc = prob.f.bcresid_prototype.x
        left_bc_size, right_bc_size = size(left_bc), size(right_bc)
        no_left_bc = length(left_bc)
    else
        left_bc = prob.f.bc[1](u0_, prob.p) # Guaranteed to be out of place here
        no_left_bc = length(left_bc)
    end

    obj = Bvpm2()
    if prob.u0 isa Function
        guess_function = @closure (x, y) -> (y .= vec(__initial_guess(prob.u0, prob.p, x)))
        bvpm2_init(obj, no_odes, no_left_bc, mesh, guess_function, eltype(u0)[],
            alg.max_num_subintervals, prob.u0)
    else
        bvpm2_init(obj, no_odes, no_left_bc, mesh, __flatten_initial_guess(prob.u0),
            eltype(u0)[], alg.max_num_subintervals)
    end

    bvp2m_f = if isinplace(prob)
        @closure (t, u, du) -> prob.f(reshape(du, u0_size), reshape(u, u0_size), prob.p, t)
    else
        @closure (t, u, du) -> du .= vec(prob.f(reshape(u, u0_size), prob.p, t))
    end
    bvp2m_bc = if isinplace(prob)
        @closure (ya, yb, bca, bcb) -> begin
            prob.f.bc[1](reshape(bca, left_bc_size), reshape(ya, u0_size), prob.p)
            prob.f.bc[2](reshape(bcb, right_bc_size), reshape(yb, u0_size), prob.p)
            return nothing
        end
    else
        @closure (ya, yb, bca, bcb) -> begin
            bca .= vec(prob.f.bc[1](reshape(ya, u0_size), prob.p))
            bcb .= vec(prob.f.bc[2](reshape(yb, u0_size), prob.p))
            return nothing
        end
    end

    opt = OptionsODE(OPT_RTOL => reltol, OPT_METHODCHOICE => alg.method_choice,
        OPT_DIAGNOSTICOUTPUT => alg.diagnostic_output,
        OPT_SINGULARTERM => alg.singular_term, OPT_ERRORCONTROL => alg.error_control)

    sol, retcode, stats = bvpm2_solve(obj, bvp2m_f!, bvp2m_bc!, opt)
    retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure

    x_mesh = bvpm2_get_x(sol)
    evalsol = evalSolution(sol, x_mesh)
    sol_final = DiffEqBase.build_solution(prob, alg, x_mesh,
        map(x -> reshape(convert(Vector{eltype(evalsol)}, x), u0_size), eachcol(evalsol));
        retcode, stats)

    bvpm2_destroy(obj)
    bvpm2_destroy(sol)

    return sol_final
end

#-------
# BVPSOL
#-------
function __solve(prob::BVProblem, alg::BVPSOL; maxiters = 1000, reltol = 1e-3, dt = 0.0,
        verbose = true, kwargs...)
    if !(prob.problem_type isa TwoPointBVProblem)
        throw(ArgumentError("`BVPSOL` only supports `TwoPointBVProblem!`"))
    end
    if !__has_initial_guess(prob.u0)
        throw(ArgumentError("Initial Guess is required for `BVPSOL`"))
    end

    t₀, t₁ = prob.tspan
    u0_ = __extract_u0(prob.u0, prob.p, t₀)
    u0_size = size(u0_)
    n = __initial_guess_length(prob.u0)

    n == -1 && dt ≤ 0 && throw(ArgumentError("`dt` must be positive."))
    u0 = __flatten_initial_guess(prob.u0)
    mesh = __extract_mesh(prob.u0, t₀, t₁, ifelse(n == -1, dt, n))
    if u0 === nothing
        # initial_guess function was provided
        u0 = mapreduce(@closure(t -> vec(__initial_guess(prob.u0, prob.p, t))), hcat, mesh)
    end

    if prob.f.bcresid_prototype !== nothing
        left_bc, right_bc = prob.f.bcresid_prototype.x
        left_bc_size, right_bc_size = size(left_bc), size(right_bc)
        no_left_bc = length(left_bc)
    else
        left_bc = prob.f.bc[1](u0_, prob.p) # Guaranteed to be out of place here
        no_left_bc = length(left_bc)
    end

    opt = OptionsODE(OPT_RTOL => reltol, OPT_MAXSTEPS => maxiters,
        OPT_BVPCLASS => alg.bvpclass, OPT_SOLMETHOD => alg.sol_method,
        OPT_RHS_CALLMODE => RHS_CALL_INSITU)

    bvpsol_f = if isinplace(prob)
        @closure (t, u, du) -> prob.f(reshape(du, u0_size), reshape(u, u0_size), prob.p, t)
    else
        @closure (t, u, du) -> du .= vec(prob.f(reshape(u, u0_size), prob.p, t))
    end

    bvpsol_bc = if isinplace(prob)
        @closure (ya, yb, r) -> begin
            left_bc = reshape(@view(r[1:no_left_bc]), left_bc_size)
            right_bc = reshape(@view(r[(no_left_bc + 1):end]), right_bc_size)
            prob.f.bc[1](left_bc, reshape(ya, u0_size), prob.p)
            prob.f.bc[2](right_bc, reshape(yb, u0_size), prob.p)
            return nothing
        end
    else
        @closure (ya, yb, r) -> begin
            r[1:no_left_bc] .= vec(prob.f.bc[1](reshape(ya, u0_size), prob.p))
            r[(no_left_bc + 1):end] .= vec(prob.f.bc[2](reshape(yb, u0_size), prob.p))
            return nothing
        end
    end

    sol_t, sol_x, retcode, stats = bvpsol(bvpsol_f, bvpsol_bc, mesh, u0, alg.odesolver, opt)

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
        map(x -> reshape(convert(Vector{eltype(evalsol)}, x), u0_size), eachcol(sol_x));
        retcode = retcode ≥ 0 ? ReturnCode.Success : ReturnCode.Failure, stats)
end

end
