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

Defect controller, with the maximum defec threshold as 0.1.
"""
struct DefectControl{T} <: AbstractErrorControl
    defect_threshold::T

    function DefectControl(; defect_threshold = 0.1)
        return new{typeof(defect_threshold)}(defect_threshold)
    end
end

"""
    GlobalErrorControl(; method = HOErorControl())

Global error controller, need to specify which gloabal error want to use.
"""
struct GlobalErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod

    function GlobalErrorControl(; method = HOErrorControl())
        return new(method)
    end
end

"""
    SequentialErrorControl(; method = HOErrorControl())

First use defect controller, if the defect is satisfying, then use global error controller.
"""
struct SequentialErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod

    function SequentialErrorControl(; method = HOErrorControl())
        return new(method)
    end
end

"""
    HybridErrorControl(; DE = 1.0, GE = 1.0; method = HOErrorControl())

Control both of the defect and global error, where the error norm is the linear combination of the defect and global error.
"""
struct HybridErrorControl{T1, T2} <: AbstractErrorControl
    DE::T1
    GE::T2
    method::GlobalErrorControlMethod

    function HybridErrorControl(; DE = 1.0, GE = 1.0, method = HOErrorControl())
        return new{typeof(DE), typeof(GE)}(DE, GE, method)
    end
end

"""
    HOErrorControl()

Higher order global error estimation method

Solve the BVP twice with an order+2 method on the original mesh
"""
struct HOErrorControl <: GlobalErrorControlMethod end

"""
    REErrorControl()

Richardson extrapolation global error estimation method

Solve the BVP twice on teh doubled mesh with the original method
"""
struct REErrorControl <: GlobalErrorControlMethod end

# Some utils for error control adaptivity
# If error control use both defect and global error or not
@inline __use_both_error_control(::SequentialErrorControl) = true
@inline __use_both_error_control(::HybridErrorControl) = true
@inline __use_both_error_control(_) = false
