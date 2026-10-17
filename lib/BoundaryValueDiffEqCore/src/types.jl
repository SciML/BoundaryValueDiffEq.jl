# Sparsity Detection
"""
    BVPJacobianAlgorithm(diffmode = missing; nonbc_diffmode = missing, bc_diffmode = missing)

Select the automatic differentiation backends used to form boundary value problem
Jacobians.

For two-point problems, `diffmode` is used for the whole residual. For standard
multi-point problems, `nonbc_diffmode` is used for the differential equation residual and
`bc_diffmode` is used for the boundary condition residual. Passing `diffmode` fills both
specialized fields unless they are supplied explicitly.
"""
@concrete struct BVPJacobianAlgorithm
    bc_diffmode
    nonbc_diffmode
    diffmode
end

"""
    __materialize_jacobian_algorithm(nlsolve, jac_alg)

Normalize a user supplied AD backend or Jacobian algorithm into a `BVPJacobianAlgorithm`.
When `jac_alg` is not supplied, the nonlinear solver's `jacobian_ad` field is used if it
exists.
"""
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
    return print(io, ")")
end

"""
    __any_sparse_ad(ad_or_jac_alg) -> Bool

Return whether an AD backend or `BVPJacobianAlgorithm` contains an `AutoSparse` backend.
"""
@inline __any_sparse_ad(::AutoSparse) = true
@inline function __any_sparse_ad(jac_alg::BVPJacobianAlgorithm)
    return __any_sparse_ad(jac_alg.bc_diffmode) ||
        __any_sparse_ad(jac_alg.nonbc_diffmode) ||
        __any_sparse_ad(jac_alg.diffmode)
end
@inline __any_sparse_ad(_) = false

function BVPJacobianAlgorithm(diffmode = missing; nonbc_diffmode = missing, bc_diffmode = missing)
    if diffmode !== missing
        bc_diffmode = bc_diffmode === missing ? diffmode : bc_diffmode
        nonbc_diffmode = nonbc_diffmode === missing ? diffmode : nonbc_diffmode
        return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
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
function concrete_jacobian_algorithm(jac_alg::BVPJacobianAlgorithm, prob::AbstractBVProblem, alg)
    return concrete_jacobian_algorithm(jac_alg, prob.problem_type, prob, alg)
end

# For multi-point BVP, we only care about bc_diffmode and nonbc_diffmode
function concrete_jacobian_algorithm(
        jac_alg::BVPJacobianAlgorithm, prob_type::StandardBVProblem, prob::BVProblem, alg
    )
    u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    bc_diffmode = jac_alg.bc_diffmode === nothing ? __default_bc_sparse_ad(u0) :
        jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? __default_sparse_ad(u0) :
        jac_alg.nonbc_diffmode
    diffmode = jac_alg.diffmode === nothing ? nothing : jac_alg.diffmode
    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

# For two-point BVP, we only care about diffmode
function concrete_jacobian_algorithm(
        jac_alg::BVPJacobianAlgorithm, prob_type::TwoPointBVProblem, prob::BVProblem, alg
    )
    u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    diffmode = jac_alg.diffmode === nothing ? __default_sparse_ad(u0) : jac_alg.diffmode
    bc_diffmode = jac_alg.bc_diffmode === nothing ? nothing : jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? nothing : jac_alg.nonbc_diffmode
    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

function concrete_jacobian_algorithm(jac_alg::BVPJacobianAlgorithm, prob_type, prob::SecondOrderBVProblem, alg)
    u0 = __extract_u0(prob.u0, prob.p, first(prob.tspan))
    diffmode = jac_alg.diffmode === nothing ? __default_sparse_ad(u0) : jac_alg.diffmode
    bc_diffmode = jac_alg.bc_diffmode === nothing ?
        (
            prob_type isa TwoPointSecondOrderBVProblem ? __default_bc_sparse_ad :
            __default_nonsparse_ad
        )(u0) : jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? __default_sparse_ad(u0) :
        jac_alg.nonbc_diffmode

    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

"""
    __default_sparse_ad(x_or_type)

Choose the default sparse AD backend for an input value or element type.
"""
@inline function __default_sparse_ad(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? __default_sparse_ad(T) : __default_sparse_ad(first(x))
end
@inline __default_sparse_ad(x::T) where {T} = __default_sparse_ad(T)
@inline __default_sparse_ad(::Type{<:Complex}) = AutoSparse(
    AutoFiniteDiff(), sparsity_detector = TracerLocalSparsityDetector(),
    coloring_algorithm = GreedyColoringAlgorithm()
)
@inline function __default_sparse_ad(::Type{T}) where {T}
    return AutoSparse(
        ifelse(ForwardDiff.can_dual(T), AutoForwardDiff(), AutoFiniteDiff()),
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    )
end

@inline function __default_bc_sparse_ad(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? __default_bc_sparse_ad(T) : __default_bc_sparse_ad(first(x))
end
@inline __default_bc_sparse_ad(x::T) where {T} = __default_bc_sparse_ad(T)
@inline __default_bc_sparse_ad(::Type{<:Complex}) = AutoSparse(
    AutoFiniteDiff(), sparsity_detector = TracerLocalSparsityDetector(),
    coloring_algorithm = GreedyColoringAlgorithm()
)
@inline function __default_bc_sparse_ad(::Type{T}) where {T}
    return AutoSparse(
        ifelse(ForwardDiff.can_dual(T), AutoForwardDiff(), AutoFiniteDiff()),
        sparsity_detector = TracerLocalSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    )
end

"""
    __default_coloring_algorithm(diffmode)

Return the sparse matrix coloring algorithm associated with `diffmode`, or the package
default when none is specified.
"""
@inline __default_coloring_algorithm(_) = GreedyColoringAlgorithm()
@inline __default_coloring_algorithm(diffmode::AutoSparse) = isnothing(diffmode) ?
    GreedyColoringAlgorithm() :
    diffmode.coloring_algorithm

"""
    __default_sparsity_detector(diffmode)

Return the sparsity detector associated with `diffmode`, or the package default when none
is specified.
"""
@inline __default_sparsity_detector(_) = TracerLocalSparsityDetector()
@inline __default_sparsity_detector(diffmode::AutoSparse) = isnothing(diffmode) ?
    TracerLocalSparsityDetector() :
    diffmode.sparsity_detector

"""
    __default_nonsparse_ad(x_or_type)

Choose the default dense AD backend for an input value or element type.
"""
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

"""
    __needs_diffcache(ad_or_jac_alg) -> Bool

Return whether the AD backend needs a `PreallocationTools.DiffCache` during residual or
Jacobian evaluation.
"""
@inline __needs_diffcache(::AutoForwardDiff) = true
@inline __needs_diffcache(::AutoPolyesterForwardDiff) = true
@inline __needs_diffcache(ad::AutoSparse) = __needs_diffcache(ADTypes.dense_ad(ad))
@inline __needs_diffcache(_) = false
@inline function __needs_diffcache(jac_alg::BVPJacobianAlgorithm)
    return __needs_diffcache(jac_alg.diffmode) ||
        __needs_diffcache(jac_alg.bc_diffmode) ||
        __needs_diffcache(jac_alg.nonbc_diffmode)
end

"""
    __maybe_allocate_diffcache(x, chunksize, jac_alg)

Allocate a `DiffCache` for `x` when `jac_alg` requires one; otherwise return `x`.
"""
function __maybe_allocate_diffcache(x, chunksize, jac_alg)
    return __needs_diffcache(jac_alg) ?
        DiffCache(x, chunksize; warn_on_resize = false) : x
end
function __maybe_allocate_diffcache(x::DiffCache, chunksize)
    return DiffCache(zero(x.du), chunksize; warn_on_resize = false)
end

# DiffCache
"""
    DiffCacheNeeded

Trait value indicating that a Jacobian backend needs a `DiffCache`.
"""
struct DiffCacheNeeded end

"""
    NoDiffCacheNeeded

Trait value indicating that a Jacobian backend does not need a `DiffCache`.
"""
struct NoDiffCacheNeeded end

"""
    __cache_trait(ad_or_jac_alg)

Return `DiffCacheNeeded()` or `NoDiffCacheNeeded()` for an AD backend or
`BVPJacobianAlgorithm`.
"""
@inline __cache_trait(::AutoForwardDiff) = DiffCacheNeeded()
@inline __cache_trait(ad::AutoSparse) = __cache_trait(ADTypes.dense_ad(ad))
@inline function __cache_trait(jac_alg::BVPJacobianAlgorithm)
    return __needs_diffcache(jac_alg) ? DiffCacheNeeded() : NoDiffCacheNeeded()
end
@inline __cache_trait(_) = NoDiffCacheNeeded()
