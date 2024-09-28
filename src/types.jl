# Sparsity Detection
@concrete struct BVPJacobianAlgorithm
    bc_diffmode
    nonbc_diffmode
    diffmode
end

@inline __materialize_jacobian_algorithm(_, alg::BVPJacobianAlgorithm) = alg
@inline __materialize_jacobian_algorithm(_, alg::ADTypes.AbstractADType) = BVPJacobianAlgorithm(alg)
@inline __materialize_jacobian_algorithm(::Nothing, ::Nothing) = BVPJacobianAlgorithm()
@inline function __materialize_jacobian_algorithm(nlsolve::N, ::Nothing) where {N}
    ad = hasfield(N, :jacobian_ad) ? nlsolve.jacobian_ad : missing
    return BVPJacobianAlgorithm(ad)
end

function Base.show(io::IO, alg::BVPJacobianAlgorithm)
    print(io, "BVPJacobianAlgorithm(")
    modifiers = String[]
    if alg.diffmode !== nothing && alg.diffmode !== missing
        push!(modifiers, "diffmode = $(__nameof(alg.diffmode))()")
    else
        if alg.nonbc_diffmode !== missing && alg.nonbc_diffmode !== nothing
            push!(modifiers, "nonbc_diffmode = $(__nameof(alg.nonbc_diffmode))()")
        end
        if alg.bc_diffmode !== missing && alg.bc_diffmode !== nothing
            push!(modifiers, "bc_diffmode = $(__nameof(alg.bc_diffmode))()")
        end
    end
    print(io, join(modifiers, ", "))
    print(io, ")")
end

@inline __any_sparse_ad(::AutoSparse) = true
@inline function __any_sparse_ad(jac_alg::BVPJacobianAlgorithm)
    __any_sparse_ad(jac_alg.bc_diffmode) ||
        __any_sparse_ad(jac_alg.nonbc_diffmode) ||
        __any_sparse_ad(jac_alg.diffmode)
end
@inline __any_sparse_ad(_) = false

function BVPJacobianAlgorithm(
        diffmode = missing; nonbc_diffmode = missing, bc_diffmode = missing)
    if diffmode !== missing
        bc_diffmode = bc_diffmode === missing ? diffmode : bc_diffmode
        nonbc_diffmode = nonbc_diffmode === missing ? diffmode : nonbc_diffmode
        return BVPJacobianAlgorithm(diffmode, diffmode, diffmode)
    else
        diffmode = nothing
        bc_diffmode = bc_diffmode === missing ? nothing : bc_diffmode
        nonbc_diffmode = nonbc_diffmode === missing ? nothing : nonbc_diffmode
        return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
    end
end

"""
    concrete_jacobian_algorithm(jac_alg, prob, alg)
    concrete_jacobian_algorithm(jac_alg, problem_type, prob, alg)

If user provided all the required fields, then return the user provided algorithm.
Otherwise, based on the problem type and the algorithm, decide the missing fields.

For example, for `TwoPointBVProblem`, the `bc_diffmode` is set to
`AutoSparse(AutoForwardDiff())` while for `StandardBVProblem`, the `bc_diffmode` is set to
`AutoForwardDiff()`.
"""
function concrete_jacobian_algorithm(jac_alg::BVPJacobianAlgorithm, prob::BVProblem, alg)
    return concrete_jacobian_algorithm(jac_alg, prob.problem_type, prob, alg)
end

function concrete_jacobian_algorithm(
        jac_alg::BVPJacobianAlgorithm, prob_type, prob::BVProblem, alg)
    u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    diffmode = jac_alg.diffmode === nothing ? __default_sparse_ad(u0) : jac_alg.diffmode
    bc_diffmode = jac_alg.bc_diffmode === nothing ?
                  (prob_type isa TwoPointBVProblem ? __default_sparse_ad :
                   __default_nonsparse_ad)(u0) : jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? __default_sparse_ad(u0) :
                     jac_alg.nonbc_diffmode

    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

@inline function __default_sparse_ad(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? __default_sparse_ad(T) : __default_sparse_ad(first(x))
end
@inline __default_sparse_ad(x::T) where {T} = __default_sparse_ad(T)
@inline __default_sparse_ad(::Type{<:Complex}) = AutoSparse(AutoFiniteDiff())
@inline function __default_sparse_ad(::Type{T}) where {T}
    return AutoSparse(ifelse(ForwardDiff.can_dual(T), AutoForwardDiff(), AutoFiniteDiff()))
end

@inline function __default_nonsparse_ad(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? __default_nonsparse_ad(T) : __default_nonsparse_ad(first(x))
end
@inline __default_nonsparse_ad(x::T) where {T} = __default_nonsparse_ad(T)
@inline __default_nonsparse_ad(::Type{<:Complex}) = AutoFiniteDiff()
@inline function __default_nonsparse_ad(::Type{T}) where {T}
    return ifelse(ForwardDiff.can_dual(T), AutoForwardDiff(), AutoFiniteDiff())
end

# This can cause Type Instability
function concretize_jacobian_algorithm(alg, prob)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return alg
end

Base.@deprecate MIRKJacobianComputationAlgorithm(
    diffmode = missing; collocation_diffmode = missing, bc_diffmode = missing) BVPJacobianAlgorithm(
    diffmode; nonbc_diffmode = collocation_diffmode, bc_diffmode)

@inline __needs_diffcache(::AutoForwardDiff) = true
@inline __needs_diffcache(ad::AutoSparse) = __needs_diffcache(ADTypes.dense_ad(ad))
@inline __needs_diffcache(_) = false
@inline function __needs_diffcache(jac_alg::BVPJacobianAlgorithm)
    return __needs_diffcache(jac_alg.diffmode) ||
           __needs_diffcache(jac_alg.bc_diffmode) ||
           __needs_diffcache(jac_alg.nonbc_diffmode)
end

# We don't need to always allocate a DiffCache. This works around that.
@concrete struct FakeDiffCache
    du
end

function __maybe_allocate_diffcache(x, chunksize, jac_alg)
    return __needs_diffcache(jac_alg) ? DiffCache(x, chunksize) : FakeDiffCache(x)
end
__maybe_allocate_diffcache(x::DiffCache, chunksize) = DiffCache(zero(x.du), chunksize)
__maybe_allocate_diffcache(x::FakeDiffCache, _) = FakeDiffCache(zero(x.du))

const MaybeDiffCache = Union{DiffCache, FakeDiffCache}

## get_tmp shows a warning as it should on cache exapansion, this behavior however is
## expected for adaptive BVP solvers so we write our own `get_tmp` and drop the warning logs
@inline get_tmp(dc::FakeDiffCache, u) = dc.du

@inline function get_tmp(dc, u)
    return Logging.with_logger(Logging.NullLogger()) do
        PreallocationTools.get_tmp(dc, u)
    end
end

# DiffCache
struct DiffCacheNeeded end
struct NoDiffCacheNeeded end

@inline __cache_trait(::AutoForwardDiff) = DiffCacheNeeded()
@inline __cache_trait(ad::AutoSparse) = __cache_trait(ADTypes.dense_ad(ad))
@inline __cache_trait(_) = NoDiffCacheNeeded()
