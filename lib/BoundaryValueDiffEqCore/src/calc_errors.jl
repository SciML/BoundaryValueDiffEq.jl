abstract type AbstractErrorControl end
abstract type GlobalErrorControlMethod end

"""
    Defect control
"""
struct DefectControl <: AbstractErrorControl end

"""
    Global error control
"""
struct GlobalErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod
end

"""
    First defect control, then global error control
"""
struct SequentialErrorControl <: AbstractErrorControl
    method::GlobalErrorControlMethod
end

"""
    Control the linear combination of defect and global error
"""
struct HybridErrorControl <: AbstractErrorControl
    DE
    GE
    method::GlobalErrorControlMethod
end

"""
    Higher order global error estimation

Solve the BVP twice with an order+2 method on the original mesh
"""
struct HOErrorControl <: GlobalErrorControlMethod end

"""
    Richardson extrapolation global error estimation

Solve the BVP twice on teh doubled mesh with the original method
"""
struct REErrorControl <: GlobalErrorControlMethod end
