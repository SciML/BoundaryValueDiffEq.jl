# Currently there are some problems with the default NonlinearSolver selection for
# BoundaryValueDiffEq
# See https://github.com/SciML/BoundaryValueDiffEq.jl/issues/175
# and https://github.com/SciML/BoundaryValueDiffEq.jl/issues/163
# These are not meant to be user facing and we should delete these once those issues are
# resolved
function __FastShortcutBVPCompatibleNLLSPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = NonlinearSolve.DEFAULT_PRECS, autodiff = nothing, kwargs...) where {T}
    if NonlinearSolve.__is_complex(T)
        algs = (GaussNewton(; concrete_jac, linsolve, precs, autodiff, kwargs...),
            LevenbergMarquardt(;
                linsolve, precs, autodiff, disable_geodesic = Val(true), kwargs...),
            LevenbergMarquardt(; linsolve, precs, autodiff, kwargs...))
    else
        algs = (GaussNewton(; concrete_jac, linsolve, precs, autodiff, kwargs...),
            LevenbergMarquardt(;
                linsolve, precs, disable_geodesic = Val(true), autodiff, kwargs...),
            TrustRegion(; concrete_jac, linsolve, precs, autodiff, kwargs...),
            GaussNewton(; concrete_jac, linsolve, precs,
                linesearch = NonlinearSolve.LineSearchesJL(; method = BackTracking()),
                autodiff, kwargs...),
            LevenbergMarquardt(; linsolve, precs, autodiff, kwargs...))
    end
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLLS))
end

function __FastShortcutBVPCompatibleNonlinearPolyalg(
        ::Type{T} = Float64; concrete_jac = nothing, linsolve = nothing,
        precs = NonlinearSolve.DEFAULT_PRECS, autodiff = nothing) where {T}
    if NonlinearSolve.__is_complex(T)
        algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),)
    else
        algs = (NewtonRaphson(; concrete_jac, linsolve, precs, autodiff),
            NewtonRaphson(; concrete_jac, linsolve, precs,
                linesearch = NonlinearSolve.LineSearchesJL(; method = BackTracking()),
                autodiff),
            TrustRegion(; concrete_jac, linsolve, precs, autodiff))
    end
    return NonlinearSolvePolyAlgorithm(algs, Val(:NLS))
end

@inline __concrete_nonlinearsolve_algorithm(prob, alg) = alg
@inline function __concrete_nonlinearsolve_algorithm(prob, ::Nothing)
    if prob isa NonlinearLeastSquaresProblem
        return __FastShortcutBVPCompatibleNLLSPolyalg(eltype(prob.u0))
    else
        return __FastShortcutBVPCompatibleNonlinearPolyalg(eltype(prob.u0))
    end
end
