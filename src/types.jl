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

# Sparsity Detection
@static if VERSION < v"1.9"
    # Sparse Linear Solvers in LinearSolve.jl are a bit flaky on older versions
    Base.@kwdef struct MIRKJacobianComputationAlgorithm{BD, CD}
        bc_diffmode::BD = AutoForwardDiff()
        collocation_diffmode::CD = AutoForwardDiff()
    end
else
    Base.@kwdef struct MIRKJacobianComputationAlgorithm{BD, CD}
        bc_diffmode::BD = AutoForwardDiff()
        collocation_diffmode::CD = AutoSparseForwardDiff()
    end
end

__needs_diffcache(::Union{AutoForwardDiff, AutoSparseForwardDiff}) = true
__needs_diffcache(_) = false
function __needs_diffcache(jac_alg::MIRKJacobianComputationAlgorithm)
    return __needs_diffcache(jac_alg.bc_diffmode) ||
           __needs_diffcache(jac_alg.collocation_diffmode)
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
