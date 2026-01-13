function BoundaryValueDiffEqCore.__resize!(x::AbstractVector{<:AbstractArray}, n, _, TU::FIRKTableau{false})
    (; s) = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N > 0 ? append!(x, [safe_similar(last(x)) for _ in 1:N]) :
        resize!(x, (n - 1) * (s + 1) + 1)
    return x
end

function BoundaryValueDiffEqCore.__resize!(x::AbstractVector{<:DiffCache}, n, M, TU::FIRKTableau{false})
    (; s) = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    if N > 0
        chunksize = isa(TU, FIRKTableau{false}) ?
            pickchunksize(M * (N + length(x) * (s + 1))) :
            pickchunksize(M * (N + length(x)))
        append!(x, [__maybe_allocate_diffcache(last(x), chunksize) for _ in 1:N])
    else
        resize!(x, (n - 1) * (s + 1) + 1)
    end
    return x
end

function BoundaryValueDiffEqCore.__resize!(x::AbstractVectorOfArray, n, M, TU::FIRKTableau{false})
    (; s) = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N > 0 ? append!(x, VectorOfArray([safe_similar(last(x)) for _ in 1:N])) :
        resize!(x, (n - 1) * (s + 1) + 1)
    return x
end

function BoundaryValueDiffEqCore.__resize!(x::AbstractVectorOfArray, n, M, TU::FIRKTableau{true})
    (; s) = TU
    N = n - length(x)
    N == 0 && return x
    N > 0 ? append!(x, VectorOfArray([safe_similar(last(x)) for _ in 1:N])) : resize!(x, n)
    return x
end
@inline __K0_on_u0(
    prob,
    stage;
    fit_parameters = false
) = __K0_on_u0(prob, prob.u0, stage; fit_parameters = fit_parameters)
@inline __K0_on_u0(
    prob,
    u0::AbstractArray,
    stage;
    fit_parameters = false) = ifelse(fit_parameters, repeat(vcat(u0, __tunable_part(prob.p)), 1, stage), repeat(u0, 1, stage))
@inline function __K0_on_u0(prob, u0::AbstractVector{<:AbstractArray}, stage; fit_parameters = false)
    u0_mat = hcat(u0...)
    avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    fit_parameters && return repeat(vcat(avg_u0, __tunable_part(prob.p)), 1, stage)
    return repeat(avg_u0, 1, stage)
end
@inline function __K0_on_u0(prob, u0::AbstractVectorOfArray, stage; fit_parameters = false)
    u0_mat = hcat(u0.u...)
    avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    fit_parameters && return repeat(vcat(avg_u0, __tunable_part(prob.p)), 1, stage)
    return repeat(avg_u0, 1, stage)
end
@inline function __K0_on_u0(prob, u0::SciMLBase.ODESolution, stage; fit_parameters = false)
    u0_mat = hcat(u0.u...)
    avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    fit_parameters && return repeat(vcat(avg_u0, __tunable_part(prob.p)), 1, stage)
    return repeat(avg_u0, 1, stage)
end
@inline function __K0_on_u0(prob, u0::F, stage; fit_parameters = false) where {
        F <:
        Function,
    }
    avg_u0 = u0(prob.p, first(prob.tspan))
    fit_parameters && return repeat(vcat(avg_u0, __tunable_part(prob.p)), 1, stage)
    return repeat(avg_u0, 1, stage)
end
