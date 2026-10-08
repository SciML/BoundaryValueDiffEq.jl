module BoundaryValueDiffEqShootingDiffEqGPUExt

using BoundaryValueDiffEqShooting: BoundaryValueDiffEqShooting as Shooting
using DiffEqGPU: DiffEqGPU, GPUODEAlgorithm
using ForwardDiff: Dual, Partials, value, partials
using KernelAbstractions: KernelAbstractions, @kernel, @index, synchronize
using SciMLBase: SciMLBase, ODEFunction, ODEProblem, ImmutableODEProblem
using StaticArrays: SVector, MVector

Shooting.__shooting_validate_ode(::GPUODEAlgorithm, platform) = nothing

# Normalize physical intervals to [0, 1] so every trajectory uses the same dt,
# while nonautonomous RHS functions still receive the physical time.
# Keep interval metadata in a concrete container. Constructing a problem inside
# a kernel must not invoke host-side parameter warnings for nested tuples.
struct ShootingParameters{P, T}
    p::P
    t0::T
    width::T
end

struct ShootingRHS{N, F, I}
    f::F
    iip::I
end
@inline function (rhs::ShootingRHS{N})(u, parameters, t) where {N}
    (; p, t0, width) = parameters
    return width * static_derivative(rhs.f, u, p, t0 + width * t, rhs.iip)
end
@inline function (rhs::ShootingRHS{N})(u, parameters, t::Dual{Tag, V, C}) where {N, Tag, V, C}
    # Time differentiation holds the state constant. Explicitly nest its dual
    # outside any shooting-state duals, avoiding tag-order-dependent promotion
    # on GPU and giving in-place RHS scratch storage the correct element type.
    T = promote_type(eltype(u), V)
    D, P = Dual{Tag, T, C}, Partials{C, T}
    state = map(x -> D(convert(T, x), zero(P)), u)
    time = D(convert(T, value(t)), convert(P, partials(t)))
    (; p, t0, width) = parameters
    return width * static_derivative(rhs.f, state, p, t0 + width * time, rhs.iip)
end

@inline function static_derivative(f::F, u::SVector{N}, p, t, ::Val{true}) where {F, N}
    du = MVector{N, eltype(u)}(undef)
    # Inlining lets the GPU compiler eliminate mutable scratch storage, including
    # when the state carries ForwardDiff partials for the sparse Jacobian.
    @inline f(du, u, p, t)
    return SVector(du)
end
@inline function static_derivative(f::F, u::SVector{N}, p, t, ::Val{false}) where {F, N}
    return SVector{N}(f(u, p, t))
end

function Shooting.__shooting_odecache(alg::GPUODEAlgorithm, u, cache, ::Type{T}) where {T}
    return shooting_odecache(u, cache, T, Val(cache.n))
end
function shooting_odecache(u, cache, ::Type{T}, ::Val{N}) where {T, N}
    Tt = eltype(cache.mesh)
    rhs = ShootingRHS{N, typeof(cache.f), typeof(cache.iip)}(cache.f, cache.iip)
    # Static parameters are prepared only during initialization. Newton residuals
    # and differentiated trajectories require no host/device state transfers.
    p = DiffEqGPU.make_static_storage(Shooting.__shooting_host(cache.p))
    f = ODEFunction{false, SciMLBase.FullSpecialize}(rhs)
    prob = ODEProblem{false}(
        f, zero(SVector{N, T}), (zero(Tt), one(Tt)), ShootingParameters(p, zero(Tt), one(Tt))
    )
    prototype = DiffEqGPU.make_prob_compatible(prob)
    isbitstype(typeof(prototype)) || throw(
        ArgumentError(
            "DiffEqGPU shooting requires a GPU-compatible RHS and static parameters."
        )
    )
    batch = KernelAbstractions.allocate(cache.platform, typeof(prototype), cache.intervals)
    return (; prob, batch, dimension = Val(N))
end

@kernel function pack_intervals!(batch, u, mesh, f::F, p, ::Val{N}) where {F, N}
    i = @index(Global, Linear)
    @inbounds begin
        state = SVector{N}(ntuple(j -> u[(i - 1) * N + j], Val(N)))
        parameters = ShootingParameters(p, mesh[i], mesh[i + 1] - mesh[i])
        batch[i] = ImmutableODEProblem{false}(
            f, state, (zero(eltype(mesh)), one(eltype(mesh))), parameters
        )
    end
end

@kernel function unpack_continuity!(r, u, values, na, ::Val{N}) where {N}
    i = @index(Global, Linear)
    @inbounds for j in 1:N
        r[na + (i - 1) * N + j] = values[end, i][j] - u[i * N + j]
    end
end

function Shooting.__shooting_integrate!(r, u, alg::GPUODEAlgorithm, cache, odecache)
    (; prob, batch, dimension) = odecache
    pack_intervals!(cache.platform)(
        batch, u, cache.mesh, prob.f, prob.p.p, dimension; ndrange = cache.intervals
    )
    synchronize(cache.platform)
    _, values = DiffEqGPU.vectorized_solve(
        batch, prob, alg; dt = one(eltype(cache.mesh)) / cache.steps,
        save_everystep = false
    )
    synchronize(cache.platform)
    unpack_continuity!(cache.platform)(
        r, u, values, cache.na, dimension; ndrange = cache.intervals
    )
    return nothing
end

end
