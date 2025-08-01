"""
    AbstractErrorControl

Abstract type for different error control methods.
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
defect = \\max\\frac{S'(x) - f(x,S(x))}{1 + |f(x,S(x))|}
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
    GlobalErrorControl(; method = HOErorControl())

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

    function HybridErrorControl(; DE = 1.0, GE = 1.0, defect = DefectControl(),
            global_error = GlobalErrorControl())
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
error = \\max\\frac{u_p - u_{p+2}}{1 + |u_p|}
```
"""
struct HOErrorControl <: GlobalErrorControlMethod end

"""
    REErrorControl()

Richardson extrapolation global error estimation method

Use Richardson extrapolation to calculate the error on the doubled mesh with

```math
error = \\frac{2^p}{2^p-1} * \\max\\frac{u_h - u_{h/2}}{1 + |u_h|}
```
"""
struct REErrorControl <: GlobalErrorControlMethod end

# Some utils for error control adaptivity
# If error control use both defect and global error or not
@inline __use_both_error_control(::HybridErrorControl) = true
@inline __use_both_error_control(_) = false
