# MIRKN Method Tableaus
struct MIRKNTableau{sType, cType, vType, wType, bType, xType, vpType, bpType, xpType}
    """Discrete stages of MIRKN formula"""
    s::sType
    c::cType
    v::vType
    w::wType
    b::bType
    x::xType
    vp::vpType
    bp::bpType
    xp::xpType

    function MIRKNTableau(s, c, v, w, b, x, vp, bp, xp)
        @assert eltype(c) ==
                eltype(v) ==
                eltype(w) ==
                eltype(b) ==
                eltype(x) ==
                eltype(vp) ==
                eltype(bp) ==
                eltype(xp)
        return new{typeof(s), typeof(c), typeof(v), typeof(w), typeof(b),
            typeof(x), typeof(vp), typeof(bp), typeof(xp)}(s, c, v, w, b, x, vp, bp, xp)
    end
end
