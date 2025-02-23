function BoundaryValueDiffEqCore.__append_similar!(
        x::AbstractVector{<:AbstractArray}, n, _, TU::FIRKTableau{false})
    (; s) = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, [similar(last(x)) for _ in 1:N])
    return x
end

function BoundaryValueDiffEqCore.__append_similar!(
        x::AbstractVector{<:MaybeDiffCache}, n, M, TU::FIRKTableau{false})
    (; s) = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    chunksize = isa(TU, FIRKTableau{false}) ? pickchunksize(M * (N + length(x) * (s + 1))) :
                pickchunksize(M * (N + length(x)))
    append!(x, [__maybe_allocate_diffcache(last(x), chunksize) for _ in 1:N])
    return x
end

function BoundaryValueDiffEqCore.__append_similar!(
        x::AbstractVectorOfArray, n, M, TU::FIRKTableau{false})
    (; s) = TU
    N = (n - 1) * (s + 1) + 1 - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, VectorOfArray([similar(last(x)) for _ in 1:N]))
    return x
end

function BoundaryValueDiffEqCore.__append_similar!(
        x::AbstractVectorOfArray, n, M, TU::FIRKTableau{true})
    (; s) = TU
    N = n - length(x)
    N == 0 && return x
    N < 0 && throw(ArgumentError("Cannot append a negative number of elements"))
    append!(x, VectorOfArray([similar(last(x)) for _ in 1:N]))
    return x
end

@inline __K0_on_u0(u0::AbstractArray, stage) = repeat(u0, 1, stage)
@inline function __K0_on_u0(u0::AbstractVector{<:AbstractArray}, stage)
    u0_mat = hcat(u0...)
    avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    return repeat(avg_u0, 1, stage)
end
@inline function __K0_on_u0(u0::AbstractVectorOfArray, stage)
    u0_mat = hcat(u0.u...)
    avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    return repeat(avg_u0, 1, stage)
end
@inline function __K0_on_u0(u0::SciMLBase.ODESolution, stage)
    u0_mat = hcat(u0.u...)
    avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
    return repeat(avg_u0, 1, stage)
end
