struct AscherTableau{ρType, cType, bType, aType}
    rho::ρType
    coef::cType
    b::bType
    acol::aType

    function AscherTableau(rho, coef, b, acol)
        return new{typeof(rho), typeof(coef), typeof(b), typeof(acol)}(rho, coef, b, acol)
    end
end

# Differential values at mesh nodes and differential derivatives/algebraic
# values at each Gauss point define the interpolated boundary solution.
struct AscherBoundarySolution{C, Z, D}
    cache::C
    z::Z
    stages::D
end

struct AscherInterpolation{S} <: SciMLBase.AbstractDiffEqInterpolation
    solution::S
end
