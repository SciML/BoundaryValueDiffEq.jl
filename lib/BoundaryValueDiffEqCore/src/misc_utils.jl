# Intermidiate solution evaluation
struct EvalSol{A <: BoundaryValueDiffEqAlgorithm}
    u
    t
    alg::A
    k_discrete
end

nodual_value(x) = x
nodual_value(x::Dual) = ForwardDiff.value(x)
nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)
nodual_value(x::AbstractArray{<:AbstractArray{<:Dual}}) = map(nodual_value, x)
