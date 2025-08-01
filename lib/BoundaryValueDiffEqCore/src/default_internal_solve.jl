# Currently there are some problems with the default NonlinearSolver selection for
# BoundaryValueDiffEq
# See https://github.com/SciML/BoundaryValueDiffEq.jl/issues/175
# and https://github.com/SciML/BoundaryValueDiffEq.jl/issues/163
# These are not meant to be user facing and we should delete these once those issues are
# resolved
function __FastShortcutBVPCompatibleNLLSPolyalg(::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, autodiff = nothing, kwargs...) where {T}
    if T <: Complex
        algs = (GaussNewton(; concrete_jac, linsolve, autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, autodiff, disable_geodesic = Val(true), kwargs...),
            LevenbergMarquardt(; linsolve, autodiff, kwargs...))
    else
        algs = (GaussNewton(; concrete_jac, linsolve, autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, disable_geodesic = Val(true), autodiff, kwargs...),
            TrustRegion(; concrete_jac, linsolve, autodiff, kwargs...),
            GaussNewton(;
                concrete_jac, linsolve, linesearch = BackTracking(), autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, autodiff, kwargs...))
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

function __FastShortcutBVPCompatibleNonlinearPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, autodiff = nothing) where {T}
    if T <: Complex
        algs = (NewtonRaphson(; concrete_jac, linsolve, autodiff),)
    else
        algs = (NewtonRaphson(; concrete_jac, linsolve, autodiff),
            NewtonRaphson(; concrete_jac, linsolve, linesearch = BackTracking(), autodiff),
            TrustRegion(; concrete_jac, linsolve, autodiff))
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

function __FastShortcutNonlinearPolyalg(::Type{T} = Float64; concrete_jac = nothing,
        linsolve = nothing, autodiff = nothing) where {T}
    if T <: Complex
        algs = (NewtonRaphson(; concrete_jac, linsolve, autodiff),)
    else
        algs = (NewtonRaphson(; concrete_jac, linsolve, autodiff),
            NewtonRaphson(; concrete_jac, linsolve, linesearch = BackTracking(), autodiff),
            TrustRegion(; concrete_jac, linsolve, autodiff))
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

"""
    __concrete_solve_algorithm(prob, nlsolve_alg, optimize_alg)

Automatic solver choosing according to the input solver.
If none of the solvers are specified, we use nonlinear solvers from NonlinearSolve.jl.
If both of the nonlinear solver and optimization solver are specified, we throw an error.
If only one of the nonlinear solver and optimization solver is specified, we use that solver.
"""
@inline __concrete_solve_algorithm(prob, alg) = alg
@inline __concrete_solve_algorithm(prob, alg, ::Nothing) = alg
@inline __concrete_solve_algorithm(prob, ::Nothing, alg) = alg
@inline __concrete_solve_algorithm(prob,
    alg1,
    alg2) = error("Both `nlsolve` and `optimize` are specified in the algorithm, but only one of them is allowed. Please specify only one of them.")
@inline function __concrete_solve_algorithm(prob, ::Nothing)
    if prob isa NonlinearLeastSquaresProblem
        return __FastShortcutBVPCompatibleNLLSPolyalg(eltype(prob.u0))
    else
        return __FastShortcutBVPCompatibleNonlinearPolyalg(eltype(prob.u0))
    end
end
@inline function __concrete_solve_algorithm(prob, ::Nothing, ::Nothing)
    if prob isa NonlinearLeastSquaresProblem
        return __FastShortcutBVPCompatibleNLLSPolyalg(eltype(prob.u0))
    else
        return __FastShortcutBVPCompatibleNonlinearPolyalg(eltype(prob.u0))
    end
end
