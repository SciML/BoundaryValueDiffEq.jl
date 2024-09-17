# FIRK Method Tableaus
struct FIRKTableau{nested, sType, aType, cType, bType}
    """Discrete stages of RK formula"""
    s::sType
    a::aType
    c::cType
    b::bType

    function FIRKTableau(s, a, c, b, nested)
        @assert eltype(a) == eltype(c) == eltype(b)
        return new{nested, typeof(s), typeof(a), typeof(c), typeof(b)}(s, a, c, b)
    end
end

struct FIRKInterpTableau{nested, c, m}
    q_coeff::c
    τ_star::m
    stage::Int

    function FIRKInterpTableau(q_coeff, τ_star, stage, nested::Bool)
        @assert eltype(q_coeff) == eltype(τ_star)
        return new{nested, typeof(q_coeff), typeof(τ_star)}(q_coeff, τ_star, stage)
    end
end
