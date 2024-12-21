# Intermidiate solution for evaluating boundry conditions
# basically simplified version of the linear interpolation for MIRKN
function (s::EvalSol{C})(tval::Number) where {C <: MIRKNCache}
    (; t, u, cache) = s

    # Quick handle for the case where tval is at the boundary
    (tval == t[1]) && return first(u)
    (tval == t[end]) && return last(u)
    # Linear interpolation
    i = interval(tval, t)
    dt = t[i + 1] - t[i]
    τ = (tval - t[i]) / dt
    z = τ * u[i + 1] + (1 - τ) * u[i]
    return z
end
