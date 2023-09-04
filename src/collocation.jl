__initial_state_from_prob(prob::BVProblem, mesh) = __initial_state_from_prob(prob.u0, mesh)
__initial_state_from_prob(u0::AbstractVector{<:Real}, mesh) = [copy(u0) for _ in mesh]
__initial_state_from_prob(u0::AbstractVector{<:AbstractVector}, _) = deepcopy(u0)
__initial_state_from_prob(u0::AbstractMatrix, _) = copy(u0)

# Auxiliary functions for evaluation
function eval_bc_residual!(residual::AbstractArray, _, bc!, y, p, mesh, u)
    return bc!(residual, y, p, mesh)
end
function eval_bc_residual!(residual::AbstractArray, ::TwoPointBVProblem, bc!, y, p, mesh, u)
    y₁ = first(y)
    y₂ = last(y)
    return bc!(residual, (y₁, y₂), p, (first(mesh), last(mesh)))
end

function Φ!(residual, cache::MIRKCache, y, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.k_discrete, cache.f!, cache.TU,
        y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ!(residual, fᵢ_cache, k_discrete, f!, TU::MIRKTableau, y, u, p,
    mesh, mesh_dt, stage::Int)
    @unpack c, v, x, b = TU

    tmp = get_tmp(fᵢ_cache, u)
    T = eltype(u)
    for i in eachindex(k_discrete)
        K = get_tmp(k_discrete[i], u)
        residᵢ = residual[i]
        h = mesh_dt[i]

        yᵢ = get_tmp(y[i], u)
        yᵢ₊₁ = get_tmp(y[i + 1], u)

        for r in 1:stage
            @. tmp = (1 - v[r]) * yᵢ + v[r] * yᵢ₊₁
            mul!(tmp, K[:, 1:(r - 1)], x[r, 1:(r - 1)], h, T(1))
            f!(K[:, r], tmp, p, mesh[i] + c[r] * h)
        end

        # Update residual
        @. residᵢ = yᵢ₊₁ - yᵢ
        mul!(residᵢ, K[:, 1:stage], b[1:stage], -h, T(1))
    end
end
