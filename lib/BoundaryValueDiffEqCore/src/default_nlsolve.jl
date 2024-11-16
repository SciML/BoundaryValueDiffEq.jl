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
            LevenbergMarquardt(;
                linsolve, autodiff, disable_geodesic = Val(true), kwargs...),
            LevenbergMarquardt(; linsolve, autodiff, kwargs...))
    else
        algs = (GaussNewton(; concrete_jac, linsolve, autodiff, kwargs...),
            LevenbergMarquardt(;
                linsolve, disable_geodesic = Val(true), autodiff, kwargs...),
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

@inline __concrete_nonlinearsolve_algorithm(prob, alg) = alg
@inline function __concrete_nonlinearsolve_algorithm(prob, ::Nothing)
    if prob isa NonlinearLeastSquaresProblem
        return __FastShortcutBVPCompatibleNLLSPolyalg(eltype(prob.u0))
    else
        return __FastShortcutBVPCompatibleNonlinearPolyalg(eltype(prob.u0))
    end
end
