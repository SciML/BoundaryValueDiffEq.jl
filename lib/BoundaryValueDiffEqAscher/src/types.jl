struct AscherTableau{ρType, cType, bType, aType}
    rho::ρType
    coef::cType
    b::bType
    acol::aType

    function AscherTableau(rho, coef, b, acol)
        return new{typeof(rho), typeof(coef), typeof(b), typeof(acol)}(rho, coef, b, acol)
    end
end
