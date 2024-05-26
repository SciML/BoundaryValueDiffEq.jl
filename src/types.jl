# MIRK Method Tableaus
struct MIRKTableau{sType, cType, vType, bType, xType}
    """Discrete stages of MIRK formula"""
    s::sType
    c::cType
    v::vType
    b::bType
    x::xType

    function MIRKTableau(s, c, v, b, x)
        @assert eltype(c) == eltype(v) == eltype(b) == eltype(x)
        return new{typeof(s), typeof(c), typeof(v), typeof(b), typeof(x)}(s, c, v, b, x)
    end
end

struct MIRKInterpTableau{s, c, v, x, τ}
    s_star::s
    c_star::c
    v_star::v
    x_star::x
    τ_star::τ

    function MIRKInterpTableau(s_star, c_star, v_star, x_star, τ_star)
        @assert eltype(c_star) == eltype(v_star) == eltype(x_star)
        return new{
            typeof(s_star), typeof(c_star), typeof(v_star), typeof(x_star), typeof(τ_star)}(
            s_star, c_star, v_star, x_star, τ_star)
    end
end

# Sparsity Detection
@concrete struct BVPJacobianAlgorithm
    bc_diffmode
    nonbc_diffmode
    diffmode
end

@inline __materialize_jacobian_algorithm(_, alg::BVPJacobianAlgorithm) = alg
@inline __materialize_jacobian_algorithm(_, alg::AbstractADType) = BVPJacobianAlgorithm(alg)
@inline __materialize_jacobian_algorithm(::Nothing, ::Nothing) = BVPJacobianAlgorithm()
# TODO: Introduce a public API in NonlinearSolve.jl for this
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
    return __any_sparse_ad(jac_alg.bc_diffmode) ||
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
    end
    diffmode = nothing
    bc_diffmode = bc_diffmode === missing ? nothing : bc_diffmode
    nonbc_diffmode = nonbc_diffmode === missing ? nothing : nonbc_diffmode
    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
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
    diffmode = jac_alg.diffmode === nothing ? __default_sparse_ad(u0) :
               __concrete_adtype(u0, jac_alg.diffmode)
    bc_diffmode = jac_alg.bc_diffmode === nothing ?
                  (prob_type isa TwoPointBVProblem ? __default_sparse_ad :
                   __default_dense_ad)(u0) : __concrete_adtype(u0, jac_alg.bc_diffmode)
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? __default_sparse_ad(u0) :
                     __concrete_adtype(u0, jac_alg.nonbc_diffmode)

    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

function concretize_jacobian_algorithm(alg, prob)
    return @set alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
end

@inline function __default_dense_ad(x::AbstractArray{T}) where {T}
    ck = __pick_nested_chunksize(x)
    isbitstype(T) && __default_dense_ad(ck, T)
    return __default_dense_ad(ck, mapreduce(eltype, promote_type, x))
end
@inline __default_dense_ad(ck::Int, ::T) where {T} = __default_dense_ad(ck, T)
@inline __default_dense_ad(_, ::Type{<:Complex}) = AutoFiniteDiff()
@inline function __default_dense_ad(ck::Int, ::Type{T}) where {T}
    return ifelse(
        ForwardDiff.can_dual(T), AutoForwardDiff(; chunksize = ck), AutoFiniteDiff())
end

@inline function __default_sparse_ad(args...)
    return AutoSparse(
        __default_dense_ad(args...); sparsity_detector = TracerSparsityDetector())
end

@inline function __concrete_adtype(u0, ad::AutoSparse)
    dense_ad = __concrete_adtype(u0, ADTypes.dense_ad(ad))
    return @set ad.dense_ad = dense_ad
end
@inline function __concrete_adtype(
        u0, ad::Union{AutoForwardDiff{nothing}, AutoPolyesterForwardDiff{nothing}})
    return parameterless_type(ad)(; chunksize = __pick_nested_chunksize(u0), tag = ad.tag)
end
@inline __concrete_adtype(_, ad::AbstractADType) = ad

@inline function __pick_nested_chunksize(x::AbstractArray{T}) where {T}
    isbitstype(T) && return DI.pick_chunksize(length(x))
    return minimum(__pick_nested_chunksize, x)
end

# DI just ignores chunksize for sparse backends
@inline __get_chunksize(ad::AutoSparse) = 1 # __get_chunksize(ADTypes.dense_ad(ad))
@inline __get_chunksize(::AutoForwardDiff{CK}) where {CK} = CK
@inline __get_chunksize(::AutoPolyesterForwardDiff{CK}) where {CK} = CK

@inline __get_tag(ad::AutoSparse) = __get_tag(ADTypes.dense_ad(ad))
@inline __get_tag(ad::Union{AutoForwardDiff, AutoPolyesterForwardDiff}) = ad.tag

# We don't need to always allocate a DiffCache. This works around that.
@concrete struct FakeDiffCache
    du
end

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

Base.:+(::DiffCacheNeeded, ::DiffCacheNeeded) = DiffCacheNeeded()
Base.:+(::NoDiffCacheNeeded, ::NoDiffCacheNeeded) = NoDiffCacheNeeded()
Base.:+(::DiffCacheNeeded, ::NoDiffCacheNeeded) = DiffCacheNeeded()
Base.:+(::NoDiffCacheNeeded, ::DiffCacheNeeded) = DiffCacheNeeded()

@inline function __cache_trait(jac_alg::BVPJacobianAlgorithm)
    return +(__cache_trait(jac_alg.diffmode), __cache_trait(jac_alg.bc_diffmode),
        __cache_trait(jac_alg.nonbc_diffmode))
end
@inline __cache_trait(::AutoForwardDiff) = DiffCacheNeeded()
@inline __cache_trait(ad::AutoSparse) = __cache_trait(ADTypes.dense_ad(ad))
@inline __cache_trait(_) = NoDiffCacheNeeded()

@inline __maybe_allocate_diffcache(x, chunksize, jac_alg) = __maybe_allocate_diffcache(
    __cache_trait(jac_alg), x, chunksize)
@inline __maybe_allocate_diffcache(::NoDiffCacheNeeded, x, chunksize) = FakeDiffCache(x)
@inline __maybe_allocate_diffcache(::DiffCacheNeeded, x, chunksize) = DiffCache(
    x, chunksize)

@inline __maybe_allocate_diffcache(x::DiffCache, chunksize) = DiffCache(
    similar(x.du), chunksize)
@inline __maybe_allocate_diffcache(x::FakeDiffCache, _) = FakeDiffCache(similar(x.du))
