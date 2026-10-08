# Currently there are some problems with the default NonlinearSolver selection for
# BoundaryValueDiffEq
# See https://github.com/SciML/BoundaryValueDiffEq.jl/issues/175
# and https://github.com/SciML/BoundaryValueDiffEq.jl/issues/163
# These are not meant to be user facing and we should delete these once those issues are
# resolved
function __FastShortcutBVPCompatibleNLLSPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, autodiff = nothing, kwargs...
    ) where {T}
    if T <: Complex
        algs = (
            GaussNewton(; concrete_jac, linsolve, autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, autodiff, disable_geodesic = Val(true), kwargs...),
            LevenbergMarquardt(; linsolve, autodiff, kwargs...),
        )
    else
        algs = (
            GaussNewton(; concrete_jac, linsolve, autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, disable_geodesic = Val(true), autodiff, kwargs...),
            TrustRegion(; concrete_jac, linsolve, autodiff, kwargs...),
            GaussNewton(;
                concrete_jac, linsolve, linesearch = BackTracking(), autodiff, kwargs...
            ),
            LevenbergMarquardt(; linsolve, autodiff, kwargs...),
        )
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

function __FastShortcutBVPCompatibleNonlinearPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, autodiff = nothing
    ) where {T}
    if T <: Complex
        algs = (NewtonRaphson(; concrete_jac, linsolve, autodiff),)
    else
        algs = (
            NewtonRaphson(; concrete_jac, linsolve, autodiff),
            NewtonRaphson(; concrete_jac, linsolve, linesearch = BackTracking(), autodiff),
            TrustRegion(; concrete_jac, linsolve, autodiff),
        )
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

"""
    __FastShortcutNonlinearPolyalg(
        T = Float64; concrete_jac = nothing, linsolve = nothing,
        autodiff = nothing
    )

Build the default nonlinear solver polyalgorithm used when a BVP algorithm does not supply
an explicit nonlinear solver.
"""
function __FastShortcutNonlinearPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, autodiff = nothing
    ) where {T}
    return __FastShortcutBVPCompatibleNonlinearPolyalg(T; concrete_jac, linsolve, autodiff)
end

"""
    __concrete_solve_algorithm(prob, nlsolve_alg, optimize_alg)
    __concrete_solve_algorithm(prob, cache::AbstractBoundaryValueDiffEqCache)

Automatic solver choosing according to the input solver.
If none of the solvers are specified, we use nonlinear solvers from NonlinearSolve.jl.
If both of the nonlinear solver and optimization solver are specified, we throw an error.
If only one of the nonlinear solver and optimization solver is specified, we use that solver.
The cache overload uses `__bvp_device_jacobian_plan(cache)` to select resident sparse
fallbacks; participating caches provide that accessor and store their BVP algorithm in `alg`.
"""
@inline __concrete_solve_algorithm(prob, alg) = alg
@inline __concrete_solve_algorithm(prob, alg, ::Nothing) = alg
@inline __concrete_solve_algorithm(prob, ::Nothing, alg) = alg
@inline __concrete_solve_algorithm(
    prob,
    alg1,
    alg2
) = error("Both `nlsolve` and `optimize` are specified in the algorithm, but only one of them is allowed. Please specify only one of them.")
@inline function __concrete_solve_algorithm(prob, ::Nothing)
    if prob isa NonlinearLeastSquaresProblem
        return __FastShortcutBVPCompatibleNLLSPolyalg(eltype(prob.u0))
    else
        return __FastShortcutBVPCompatibleNonlinearPolyalg(eltype(prob.u0))
    end
end
@inline __concrete_solve_algorithm(prob, ::Nothing, ::Nothing) =
    __concrete_solve_algorithm(prob, nothing)

"""
    __internal_solve(prob, alg; kwargs...)

Dispatch to the appropriate solve entry point for internal nonlinear, nonlinear least
squares, and optimization problems.
"""
@inline __internal_solve(
    prob::Union{SciMLBase.NonlinearProblem, SciMLBase.NonlinearLeastSquaresProblem},
    alg; kwargs...
) = __solve(prob, alg; kwargs...)
@inline __internal_solve(
    prob::SciMLBase.OptimizationProblem, alg;
    kwargs...
) = OptimizationBase.solve(prob, alg; kwargs...)

"""
    __concrete_device_solve_algorithm(prob, nlsolve, optimize;
        linsolve = nothing, concrete_jac = nothing, linesearch_fallback = false)

Select the solver for a resident sparse system, preserving an explicitly supplied
nonlinear or optimization solver. Defaults use Newton for square systems and
Gauss–Newton with LSMR for least squares. An optional backtracking fallback avoids
the augmented damping matrices unsupported by sparse device arrays.
"""
function __concrete_device_solve_algorithm(
        prob, nlsolve, optimize;
        linsolve = nothing, concrete_jac = nothing, linesearch_fallback = false
    )
    if nlsolve !== nothing || optimize !== nothing
        return __concrete_solve_algorithm(prob, nlsolve, optimize)
    end
    least_squares = prob isa NonlinearLeastSquaresProblem
    if least_squares && linsolve === nothing
        linsolve = KrylovJL_LSMR()
    end
    constructor = least_squares ? GaussNewton : NewtonRaphson
    algorithm = constructor(; concrete_jac, linsolve)
    linesearch_fallback || return algorithm
    return NonlinearSolvePolyAlgorithm(
        (
            algorithm,
            constructor(; concrete_jac, linsolve, linesearch = BackTracking()),
        )
    )
end

"""
    __device_square_linsolve(cache)

Select the linear solver for a sparse resident Newton solve by cache type;
`nothing` uses its default. Algorithms may specialize this hook for their storage.
"""
__device_square_linsolve(cache) = nothing

"""
    __device_nlls_linsolve(cache)

Select LSMR for a resident rectangular collocation system without forming dense
normal equations. Algorithms may specialize this hook when needed.
"""
__device_nlls_linsolve(cache) = KrylovJL_LSMR()

@inline function __concrete_solve_algorithm(prob, cache::AbstractBoundaryValueDiffEqCache)
    return __concrete_solve_algorithm(prob, cache, __bvp_device_jacobian_plan(cache))
end
@inline __concrete_solve_algorithm(prob, cache::AbstractBoundaryValueDiffEqCache, ::Nothing) =
    __concrete_solve_algorithm(prob, cache.alg.nlsolve, cache.alg.optimize)

function __concrete_solve_algorithm(
        nlprob, cache::AbstractBoundaryValueDiffEqCache, ::SparseJacobianCache
    )
    if cache.alg.nlsolve === nothing && cache.alg.optimize === nothing &&
            !(cache.alg.platform isa CPU)
        return __concrete_device_solve_algorithm(
            nlprob, nothing, nothing;
            linsolve = __device_square_linsolve(cache), linesearch_fallback = true
        )
    end
    return __concrete_solve_algorithm(nlprob, cache.alg.nlsolve, cache.alg.optimize)
end
function __concrete_solve_algorithm(
        nlprob::NonlinearLeastSquaresProblem, cache::AbstractBoundaryValueDiffEqCache,
        ::SparseJacobianCache
    )
    if cache.alg.nlsolve === nothing && cache.alg.optimize === nothing
        linsolve = __device_nlls_linsolve(cache)
        if !(cache.alg.platform isa CPU)
            # A stalled least-squares minimum must not trigger LM/More damping,
            # which builds unsupported CUDA sparse diagonal blocks.
            return __concrete_device_solve_algorithm(
                nlprob, nothing, nothing; linsolve, linesearch_fallback = true
            )
        end
        return __FastShortcutBVPCompatibleNLLSPolyalg(eltype(nlprob.u0); linsolve)
    end
    return __concrete_solve_algorithm(nlprob, cache.alg.nlsolve, cache.alg.optimize)
end

"""
    __needs_sparse_damping(algorithm, u)

Whether a nonlinear solver may append a damping block to its Jacobian. Such
solvers need resizable sparse storage rather than a fixed-bandwidth matrix.
"""
function __needs_sparse_damping(::Nothing, u)
    # Real-valued defaults include TrustRegion; older NonlinearSolve versions
    # use dogleg and do not need an augmented system.
    return eltype(u) <: Real && isdefined(NonlinearSolveBase, :MoreTrustRegionDescent)
end
function __needs_sparse_damping(alg::NonlinearSolvePolyAlgorithm, u)
    return any(a -> __needs_sparse_damping(a, u), alg.algs)
end
function __needs_sparse_damping(alg, u)
    @static if isdefined(NonlinearSolveBase, :MoreTrustRegionDescent)
        return hasproperty(alg, :descent) &&
            alg.descent isa NonlinearSolveBase.MoreTrustRegionDescent
    else
        return false
    end
end

"""
    __concrete_kwargs(nlsolve, optimize, nlsolve_kwargs, optimize_kwargs[, bvp_verbose])

Select and normalize the keyword arguments forwarded to the active internal nonlinear or
optimization solver.
"""
@inline __concrete_kwargs(nlsolve, ::Nothing, nlsolve_kwargs, optimize_kwargs) = (;
    nlsolve_kwargs...,
)
@inline __concrete_kwargs(::Nothing, optimize, nlsolve_kwargs, optimize_kwargs) = (;) # Doesn't support for now
@inline __concrete_kwargs(::Nothing, ::Nothing, nlsolve_kwargs, optimize_kwargs) = (;
    nlsolve_kwargs...,
)

# Overloads that handle BVP verbosity → NonlinearSolve verbosity conversion
@inline function __concrete_kwargs(
        nlsolve, ::Nothing, nlsolve_kwargs, optimize_kwargs, bvp_verbose::BVPVerbosity
    )
    return __solver_kwargs(nlsolve_kwargs, bvp_verbose.nonlinear_verbosity, NonlinearVerbosity)
end

@inline function __concrete_kwargs(
        ::Nothing, ::Nothing, nlsolve_kwargs, optimize_kwargs, bvp_verbose::BVPVerbosity
    )
    return __solver_kwargs(nlsolve_kwargs, bvp_verbose.nonlinear_verbosity, NonlinearVerbosity)
end

@inline function __concrete_kwargs(
        ::Nothing, optimize, nlsolve_kwargs, optimize_kwargs, bvp_verbose::BVPVerbosity
    )
    return __solver_kwargs(optimize_kwargs, bvp_verbose.optimization_verbosity, OptimizationVerbosity)
end

@inline function __solver_kwargs(kwargs, verbose, ::Type{V}) where {V}
    haskey(kwargs, :verbose) && return (; kwargs...)
    return (; verbose = verbose isa V ? verbose : V(verbose), kwargs...)
end
