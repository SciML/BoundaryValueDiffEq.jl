# Intermediate solution for evaluating boundary conditions
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

# Allocation-free linear solution evaluator used inside boundary kernels and host
# sparsity tracing. This has the same interpolation semantics as CPU MIRKN.
struct MIRKNDeviceEvalSol{Y, T, S}
    y::Y
    t::T
    in_size::S
    offset::Int
end

Base.size(sol::MIRKNDeviceEvalSol) = (sol.in_size..., size(sol.y, 2))
Base.size(sol::MIRKNDeviceEvalSol, d::Int) = size(sol)[d]
Base.firstindex(::MIRKNDeviceEvalSol, d::Int = 1) = 1
Base.lastindex(sol::MIRKNDeviceEvalSol) = size(sol.y, 2)
Base.lastindex(sol::MIRKNDeviceEvalSol, d::Int) = size(sol, d)
Base.length(sol::MIRKNDeviceEvalSol) = size(sol.y, 2)
Base.@propagate_inbounds function Base.getindex(sol::MIRKNDeviceEvalSol, node::Int)
    rows = (sol.offset + 1):(sol.offset + prod(sol.in_size))
    return __device_reshape(view(sol.y, rows, node), sol.in_size)
end
Base.@propagate_inbounds Base.getindex(sol::MIRKNDeviceEvalSol, ::Colon, node::Int) = sol[node]
Base.@propagate_inbounds Base.getindex(sol::MIRKNDeviceEvalSol, j::Int, node::Int) = sol.y[sol.offset + j, node]
Base.@propagate_inbounds function Base.getindex(sol::MIRKNDeviceEvalSol, indices::Vararg{Int, N}) where {N}
    @boundscheck N == length(sol.in_size) + 1 || throw(BoundsError(sol, indices))
    row, stride = indices[1], sol.in_size[1]
    for d in 2:(N - 1)
        row += (indices[d] - 1) * stride
        stride *= sol.in_size[d]
    end
    return sol.y[sol.offset + row, indices[N]]
end
Base.@propagate_inbounds Base.getindex(sol::MIRKNDeviceEvalSol, ::Colon, ::Colon, node::Int) = sol[node]
@inline Base.getproperty(sol::MIRKNDeviceEvalSol, name::Symbol) =
    name === :u ? sol : getfield(sol, name)
@inline Base.iterate(sol::MIRKNDeviceEvalSol, i::Int = 1) =
    i > length(sol) ? nothing : (sol[i], i + 1)

# Binary search is callable on device meshes; no host indexing is needed.
@inline function __mirkn_device_interval(t, mesh)
    lo, hi = 1, length(mesh)
    @inbounds while lo + 1 < hi
        mid = (lo + hi) ÷ 2
        if mesh[mid] <= t
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

struct MIRKNDeviceInterpolatedArray{T, N, S, W} <: AbstractArray{T, N}
    sol::S
    interval::Int
    weight::W
end
Base.size(u::MIRKNDeviceInterpolatedArray) = u.sol.in_size
Base.IndexStyle(::Type{<:MIRKNDeviceInterpolatedArray}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(u::MIRKNDeviceInterpolatedArray, j::Int)
    (; sol, interval, weight) = u
    # Endpoint branches also keep traced endpoint patterns exact.
    iszero(weight) && return sol.y[sol.offset + j, interval]
    isone(weight) && return sol.y[sol.offset + j, interval + 1]
    return (1 - weight) * sol.y[sol.offset + j, interval] +
        weight * sol.y[sol.offset + j, interval + 1]
end
@inline function (sol::MIRKNDeviceEvalSol)(t::Number)
    i = __mirkn_device_interval(t, sol.t)
    @inbounds weight = (t - sol.t[i]) / (sol.t[i + 1] - sol.t[i])
    return MIRKNDeviceInterpolatedArray{eltype(sol.y), length(sol.in_size), typeof(sol), typeof(weight)}(sol, i, weight)
end

# The public solution materializes interpolation results with device kernels.
struct MIRKNDeviceInterpolation{Y, T, S, P} <: SciMLBase.AbstractDiffEqInterpolation
    y::Y
    t::T
    in_size::S
    platform::P
end
SciMLBase.interp_summary(::MIRKNDeviceInterpolation) = "MIRKN device linear interpolation"

@inline __mirkn_device_output_index(::Nothing, j) = j
@inline __mirkn_device_output_index(idxs::Integer, j) = idxs
@inline __mirkn_device_output_index(idxs, j) = @inbounds idxs[j]

@kernel function __mirkn_device_interpolate_kernel!(out, y, mesh, t, idxs, ::Val{D}) where {D}
    j = @index(Global, Linear)
    row = __mirkn_device_output_index(idxs, j)
    i = __mirkn_device_interval(t, mesh)
    @inbounds begin
        h = mesh[i + 1] - mesh[i]
        w = (t - mesh[i]) / h
        out[j] = D == 0 ? (1 - w) * y[row, i] + w * y[row, i + 1] :
            (y[row, i + 1] - y[row, i]) / h
    end
end

function (id::MIRKNDeviceInterpolation)(t::Number, idxs, ::Type{Val{D}}, p, continuity::Symbol = :left) where {D}
    D in (0, 1) || throw(ArgumentError("MIRKN linear interpolation supports derivative orders zero and one."))
    n = size(id.y, 1)
    if idxs isa Integer
        1 <= idxs <= n || throw(BoundsError(Base.OneTo(n), idxs))
    elseif idxs !== nothing
        all(i -> i isa Integer && 1 <= i <= n, idxs) || throw(BoundsError(Base.OneTo(n), idxs))
    end
    out = similar(id.y, idxs === nothing ? n : (idxs isa Integer ? 1 : length(idxs)))
    device_idxs = idxs isa AbstractArray ? __device_parameter(id.platform, idxs) : idxs
    __mirkn_device_interpolate_kernel!(id.platform)(out, id.y, id.t, t, device_idxs, Val(D); ndrange = length(out))
    synchronize(id.platform)
    idxs isa Integer && return sum(out)
    idxs !== nothing && return out
    M = n ÷ 2
    return ArrayPartition(reshape(copy(view(out, 1:M)), id.in_size), reshape(copy(view(out, (M + 1):n)), id.in_size))
end
function (id::MIRKNDeviceInterpolation)(ts, idxs, deriv::Type{Val{D}}, p, continuity::Symbol = :left) where {D}
    times = collect(ts)
    return RecursiveArrayTools.DiffEqArray([id(t, idxs, deriv, p, continuity) for t in times], times)
end
function (id::MIRKNDeviceInterpolation)(
        out::AbstractArray, t, idxs::Union{Nothing, Integer, AbstractArray, Tuple},
        deriv::Type{Val{D}}, p, continuity::Symbol = :left
    ) where {D}
    copyto!(out, id(t, idxs, deriv, p, continuity))
    return out
end
