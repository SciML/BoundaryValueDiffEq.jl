# Intermidiate solution evaluation
@concrete struct EvalSol{C}
    u
    t
    cache::C
end

# Basic utilities for EvalSol

Base.size(e::EvalSol) = (size(e.u[1])..., length(e.u))
Base.size(e::EvalSol, i) = size(e)[i]

Base.axes(e::EvalSol) = Base.OneTo.(size(e))
Base.axes(e::EvalSol, d::Int) = Base.OneTo.(size(e)[d])

Base.getindex(e::EvalSol, args...) = Base.getindex(VectorOfArray(e.u), args...)
Base.eachindex(e::EvalSol) = Base.eachindex(e.u)

Base.first(e::EvalSol) = first(e.u)
Base.last(e::EvalSol) = last(e.u)

Base.firstindex(e::EvalSol) = 1
Base.lastindex(e::EvalSol) = length(e.u)

Base.length(e::EvalSol) = length(e.u)

Base.eltype(e::EvalSol) = eltype(e.u)

function Base.show(io::IO, m::MIME"text/plain", e::EvalSol)
    (print(io, "t: "); show(io, m, e.t); println(io); print(io, "u: "); show(io, m, e.u))
end
