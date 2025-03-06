abstract type AbstractErrorControl end
abstract type GlobalErrorControlMethod end

"""
    DefectControl()
"""
struct DefectControl{T} <: AbstractErrorControl
    defect_threshold::T

    function DefectControl(; defect_threshold = 0.1)
        return new{typeof(defect_threshold)}(defect_threshold)
    end
end

"""
    GlobalErrorControl(; method = HOErorControl())

Global error control methods, need to specify which gloabal error want to use.
"""
struct GlobalErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod

    function GlobalErrorControl(; method = HOErrorControl())
        return new(method)
    end
end

"""
    SequentialErrorControl()

First use defect control, if the defect is satisfying, then use global error control.
"""
struct SequentialErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod

    function SequentialErrorControl(; method = HOErrorControl())
        return new(method)
    end
end

"""
    HybridErrorControl(; DE = 1.0, GE = 1.0; method = HOErrorControl())

Control the both of the defect and global error, where the error norm is the linear combination of the defect and global error.
"""
struct HybridErrorControl{T1, T2} <: AbstractErrorControl
    DE::T1
    GE::T2
    method::GlobalErrorControlMethod

    function HybridErrorControl(; DE = 1.0, GE = 1.0, method = HOErrorControl())
        return new{typeof(DE), typeof{GE}}(DE, GE, method)
    end
end

"""
    HOErrorControl()

Higher order global error estimation

Solve the BVP twice with an order+2 method on the original mesh
"""
struct HOErrorControl <: GlobalErrorControlMethod end

"""
    REErrorControl()

Richardson extrapolation global error estimation

Solve the BVP twice on teh doubled mesh with the original method
"""
struct REErrorControl <: GlobalErrorControlMethod end

# Some utils for error control adaptivity
# If error control use both defect and global error or not
@inline __use_both_error_control(::SequentialErrorControl) = true
@inline __use_both_error_control(::HybridErrorControl) = true
@inline __use_both_error_control(_) = false
