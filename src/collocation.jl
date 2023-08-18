function BVPSystem(prob::BVProblem, mesh, alg::AbstractMIRK)
    tmp = similar(prob.u0, length(prob.u0))
    return BVPSystem(alg_order(alg), alg_stage(alg), length(prob.u0), length(mesh),
        prob.f, prob.bc, DiffCache(tmp))
end

__initial_state_from_prob(prob::BVProblem, mesh) = __initial_state_from_prob(prob.u0, mesh)
function __initial_state_from_prob(u0::AbstractVector{<:Real}, mesh)
    repeat(u0, outer = (1, length(mesh)))
end
__initial_state_from_prob(u0::AbstractVector{<:AbstractVector}, _) = reduce(hcat, u0)
__initial_state_from_prob(u0::AbstractMatrix, _) = copy(u0)

# Auxiliary functions for evaluation
@inline @views function eval_bc_residual!(residual::AbstractVector,
    ::SciMLBase.StandardBVProblem, S::BVPSystem, y, p, mesh)
    @static if VERSION ≥ v"1.9"
        y_ = eachcol(y) # Returns ColumnSlices which can be indexed into
    else
        y_ = collect(eachcol(y)) # Can't index into Generator
    end
    S.bc!(residual, y_, p, mesh)
end
@inline @views function eval_bc_residual!(residual::AbstractVector, ::TwoPointBVProblem, y,
    S::BVPSystem, p, mesh)
    S.bc!(residual, (y[:, 1], y[:, end]), p, (mesh[1], mesh[end]))
end

@views function Φ!(residual::AbstractMatrix, S::BVPSystem, TU::MIRKTableau,
    cache::AbstractMIRKCache, y::AbstractMatrix, p, mesh)
    @unpack M, N, f!, stage = S
    @unpack c, v, x, b = TU

    tmp = get_tmp(S.tmp, y)
    k_discrete = get_tmp(cache.k_discrete, y)

    T = eltype(y)
    for i in 1:(N - 1)
        K = k_discrete[:, :, i]
        h = mesh[i + 1] - mesh[i]

        for r in 1:stage
            @. tmp = (1 - v[r]) * y[:, i] + v[r] * y[:, i + 1]
            mul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            f!(K[:, r], tmp, p, mesh[i] + c[r] * h)
        end

        # Update residual
        @. residual[:, i] = y[:, i + 1] - y[:, i]
        mul!(residual[:, i], K[:, 1:stage], b[1:stage], -h, T(1))
    end
end
