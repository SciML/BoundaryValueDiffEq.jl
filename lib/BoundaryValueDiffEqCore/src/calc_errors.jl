"""
    AbstractErrorControl

Developer-facing abstract type for error-controller tags used by BoundaryValueDiffEq solver
implementations.

This is a narrow versioned interface for solver packages. Subtypes classify how a solver's own
adaptivity implementation manages error estimates; subtyping this type alone does not make a
controller usable by `MIRK`, `FIRK`, or another concrete solver.

# Extension Rules

A solver package that owns both an error controller and the corresponding adaptivity behavior may
subtype `AbstractErrorControl`. It may extend [`__use_both_error_control`](@ref) for that subtype
to declare whether its cache requires separate defect and global-error storage. The method must
return a `Bool`, be side-effect free, and be defined only for the extending package's controller
type. The default is `false`.

The concrete solver package remains responsible for implementing all controller-specific error
estimation and mesh-selection behavior. Applications should use the documented concrete
controllers rather than subtype this type.

# Examples

```julia
struct MyCombinedControl <: AbstractErrorControl end

BoundaryValueDiffEqCore.__use_both_error_control(::MyCombinedControl) = true
```
"""
abstract type AbstractErrorControl end

"""
    GlobalErrorControlMethod

Abstract type for different global error control methods, and according to the different global error estimation methods, there are

  - `HOErrorControl`: Higher order global error estimation method
  - `REErrorControl`: Richardson extrapolation global error estimation method
"""
abstract type GlobalErrorControlMethod end

"""
    DefectControl(; defect_threshold = 0.1)

Defect estimation method with defect defined as

```math
\\text{defect} = \\max\\frac{S'(x) - f(x,S(x))}{1 + |f(x,S(x))|}
```

Defect controller, with the maximum `defect_threshold` as 0.1, when the estimating defect is greater than the `defect_threshold`, the mesh will be refined.
"""
struct DefectControl{T} <: AbstractErrorControl
    defect_threshold::T

    function DefectControl(; defect_threshold = 0.1)
        return new{typeof(defect_threshold)}(defect_threshold)
    end
end

"""
    GlobalErrorControl(; method = HOErrorControl())

Global error controller, use high order global error estimation method `HOErrorControl` as default.
"""
struct GlobalErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod

    function GlobalErrorControl(; method = HOErrorControl())
        return new(method)
    end
end

"""
    SequentialErrorControl(; defect = DefectControl(), global_error = GlobalErrorControl())

First use the defect controller, if the defect is satisfying, then use global error controller.
"""
struct SequentialErrorControl <: AbstractErrorControl
    defect::DefectControl
    global_error::GlobalErrorControl

    function SequentialErrorControl(; defect = DefectControl(), global_error = GlobalErrorControl())
        return new(defect, global_error)
    end
end

"""
    HybridErrorControl(; DE = 1.0, GE = 1.0, defect = DefectControl(), global_error = GlobalErrorControl())

Control both of the defect and global error, where the error norm is the linear combination of the defect and global error.
"""
struct HybridErrorControl{T1, T2} <: AbstractErrorControl
    DE::T1
    GE::T2
    defect::DefectControl
    global_error::GlobalErrorControl

    function HybridErrorControl(;
            DE = 1.0, GE = 1.0, defect = DefectControl(),
            global_error = GlobalErrorControl()
        )
        return new{typeof(DE), typeof(GE)}(DE, GE, defect, global_error)
    end
end

"""
    NoErrorControl()

No error control method.
"""
struct NoErrorControl <: AbstractErrorControl end

"""
    HOErrorControl()

Higher order global error estimation method

Uses a solution from order+2 method on the original mesh and calculate the error with

```math
\\text{error} = \\max\\frac{u_p - u_{p+2}}{1 + |u_p|}
```
"""
struct HOErrorControl <: GlobalErrorControlMethod end

"""
    REErrorControl()

Richardson extrapolation global error estimation method

Use Richardson extrapolation to calculate the error on the doubled mesh with

```math
\\text{error} = \\frac{2^p}{2^p-1} ⋅ \\max\\frac{u_h - u_{h/2}}{1 + |u_h|}
```
"""
struct REErrorControl <: GlobalErrorControlMethod end

# Some utils for error control adaptivity
# If error control use both defect and global error or not
"""
    __use_both_error_control(controller) -> Bool

Return whether an error controller requires separate defect and global-error storage.

This developer hook is used while a solver cache is constructed. The default implementation
returns `false`. Solver packages may extend it only for their own
[`AbstractErrorControl`](@ref) subtype, return a concrete `Bool`, and perform no mutation. A
`true` result reserves storage for both estimates; it does not by itself add support for a custom
controller to a concrete solver.

# Examples

```julia
struct MyCombinedControl <: AbstractErrorControl end

BoundaryValueDiffEqCore.__use_both_error_control(::MyCombinedControl) = true
```
"""
@inline __use_both_error_control(::HybridErrorControl) = true
@inline __use_both_error_control(_) = false
