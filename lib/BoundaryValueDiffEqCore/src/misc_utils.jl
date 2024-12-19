# Intermidiate solution evaluation
@concrete struct EvalSol{C}
    u
    t
    cache::C
end

Base.size(e::EvalSol) = (size(e.u[1])..., length(e.u))
Base.size(e::EvalSol, i) = size(e)[i]

Base.axes(e::EvalSol) = Base.OneTo.(size(e))
Base.axes(e::EvalSol, i) = Base.OneTo.(size(e)[d])

Base.getindex(e::EvalSol, args...) = Base.getindex(VectorOfArray(e.u), args...)
