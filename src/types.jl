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

struct RKInterpTableau{s, a, c, τ}
    s_star::s
    a_star::a
    c_star::c
    τ_star::τ

    function RKInterpTableau(s_star, a_star, c_star, τ_star)
        @assert eltype(a_star) == eltype(c_star)
        return new{typeof(s_star), typeof(a_star), typeof(c_star),
            typeof(τ_star)}(s_star,
            a_star, c_star, τ_star)
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
        @assert nonbc_diffmode === missing && bc_diffmode === missing
        return BVPJacobianAlgorithm(diffmode, diffmode, diffmode)
    else
        diffmode = AutoSparseForwardDiff()
        bc_diffmode = bc_diffmode === missing ? AutoForwardDiff() : bc_diffmode
        nonbc_diffmode = nonbc_diffmode === missing ?
                         AutoSparseForwardDiff() : nonbc_diffmode
        return BVPJacobianAlgorithm(bc_diffmode, nonbc_diffmode, nonbc_diffmode)
    end
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

function maybe_allocate_diffcache(x, chunksize, jac_alg)
    if __needs_diffcache(jac_alg)
        return DiffCache(x, chunksize)
    else
        return FakeDiffCache(x)
    end
end
maybe_allocate_diffcache(x::DiffCache, chunksize) = DiffCache(similar(x.du), chunksize)
maybe_allocate_diffcache(x::FakeDiffCache, _) = FakeDiffCache(similar(x.du))

PreallocationTools.get_tmp(dc::FakeDiffCache, _) = dc.du

const MaybeDiffCache = Union{DiffCache, FakeDiffCache}
