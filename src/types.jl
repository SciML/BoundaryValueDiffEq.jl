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

@truncate_stacktrace MIRKTableau 1

struct MIRKInterpTableau{s, c, v, x, τ}
    s_star::s
    c_star::c
    v_star::v
    x_star::x
    τ_star::τ

    function MIRKInterpTableau(s_star, c_star, v_star, x_star, τ_star)
        @assert eltype(c_star) == eltype(v_star) == eltype(x_star)
        return new{typeof(s_star), typeof(c_star), typeof(v_star), typeof(x_star),
            typeof(τ_star)}(s_star,
            c_star, v_star, x_star, τ_star)
    end
end

@truncate_stacktrace MIRKInterpTableau 1

# RK Method Tableaus
struct RKTableau{nested, sType, aType, cType, bType}
    """Discrete stages of RK formula"""
    s::sType
    a::aType
    c::cType
    b::bType

    function RKTableau(s, a, c, b, nested)
        @assert eltype(a) == eltype(c) == eltype(b)
        return new{nested, typeof(s), typeof(a), typeof(c), typeof(b)}(s, a, c, b)
    end
end

@truncate_stacktrace RKTableau 1

struct RKInterpTableau{c, m, d}
    poly_coeffs::c
    poly_max::m
    dn_coeffs::d

    function RKInterpTableau(poly_coeffs, poly_max, dn_coeffs)
        @assert eltype(poly_coeffs) == eltype(poly_max) == eltype(dn_coeffs)
        return new{typeof(poly_coeffs), typeof(poly_max)}(poly_coeffs,
            poly_max, dn_coeffs)
    end
end

@truncate_stacktrace RKInterpTableau 1

# Sparsity Detection
@concrete struct BVPJacobianAlgorithm
    bc_diffmode
    nonbc_diffmode
    diffmode
end

function BVPJacobianAlgorithm(diffmode = missing; nonbc_diffmode = missing,
    bc_diffmode = missing)
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
`AutoSparseForwardDiff` while for `StandardBVProblem`, the `bc_diffmode` is set to
`AutoForwardDiff`.
"""
function concrete_jacobian_algorithm(jac_alg::BVPJacobianAlgorithm, prob::BVProblem, alg)
    return concrete_jacobian_algorithm(jac_alg, prob.problem_type, prob, alg)
end

function concrete_jacobian_algorithm(jac_alg::BVPJacobianAlgorithm, ::StandardBVProblem,
    prob::BVProblem, alg)
    diffmode = jac_alg.diffmode === nothing ? AutoSparseForwardDiff() : jac_alg.diffmode
    bc_diffmode = jac_alg.bc_diffmode === nothing ? AutoForwardDiff() : jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? AutoSparseForwardDiff() :
                     jac_alg.nonbc_diffmode

    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

function concrete_jacobian_algorithm(jac_alg::BVPJacobianAlgorithm, ::TwoPointBVProblem,
    prob::BVProblem, alg)
    diffmode = jac_alg.diffmode === nothing ? AutoSparseForwardDiff() : jac_alg.diffmode
    bc_diffmode = jac_alg.bc_diffmode === nothing ? AutoSparseForwardDiff() :
                  jac_alg.bc_diffmode
    nonbc_diffmode = jac_alg.nonbc_diffmode === nothing ? AutoSparseForwardDiff() :
                     jac_alg.nonbc_diffmode

    return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, diffmode)
end

# This can cause Type Instability
function concretize_jacobian_algorithm(alg, prob)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return alg
end

function MIRKJacobianComputationAlgorithm(diffmode = missing;
    collocation_diffmode = missing, bc_diffmode = missing)
    Base.depwarn("`MIRKJacobianComputationAlgorithm` has been deprecated in favor of \
        `BVPJacobianAlgorithm`. Replace `collocation_diffmode` with `nonbc_diffmode",
        :MIRKJacobianComputationAlgorithm)
    return BVPJacobianAlgorithm(diffmode; nonbc_diffmode = collocation_diffmode,
        bc_diffmode)
end

__needs_diffcache(::Union{AutoForwardDiff, AutoSparseForwardDiff}) = true
__needs_diffcache(_) = false
function __needs_diffcache(jac_alg::BVPJacobianAlgorithm)
    return __needs_diffcache(jac_alg.diffmode) || __needs_diffcache(jac_alg.bc_diffmode) ||
           __needs_diffcache(jac_alg.nonbc_diffmode)
end

# We don't need to always allocate a DiffCache. This works around that.
@concrete struct FakeDiffCache
    du
end

function __maybe_allocate_diffcache(x, chunksize, jac_alg)
    return __needs_diffcache(jac_alg) ? DiffCache(x, chunksize) : FakeDiffCache(x)
end
__maybe_allocate_diffcache(x::DiffCache, chunksize) = DiffCache(similar(x.du), chunksize)
__maybe_allocate_diffcache(x::FakeDiffCache, _) = FakeDiffCache(similar(x.du))

PreallocationTools.get_tmp(dc::FakeDiffCache, _) = dc.du

const MaybeDiffCache = Union{DiffCache, FakeDiffCache}
