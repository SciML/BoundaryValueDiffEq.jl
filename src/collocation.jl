__initial_state_from_prob(prob::BVProblem, mesh) = __initial_state_from_prob(prob.u0, mesh)
__initial_state_from_prob(u0::AbstractVector{<:Real}, mesh) = [copy(u0) for _ in mesh]
__initial_state_from_prob(u0::AbstractVector{<:AbstractVector}, _) = deepcopy(u0)
__initial_state_from_prob(u0::AbstractMatrix, _) = copy(u0)

# Auxiliary functions for evaluation
function eval_bc_residual!(residual::DiffCache, _, bc!, y, p, mesh, u)
    resid = get_tmp(residual, u)
    ys = [get_tmp(yᵢ, u) for yᵢ in y]
    bc!(resid, ys, p, mesh)
end
function eval_bc_residual!(residual::DiffCache, ::TwoPointBVProblem, bc!, y, p, mesh, u)
    resid = get_tmp(residual, u)
    y₁ = get_tmp(first(y), u)
    y₂ = get_tmp(last(y), u)
    bc!(resid, (y₁, y₂), p, (first(mesh), last(mesh)))
end

@views Φ!(cache::MIRKCache, u, p = cache.p) = Φ!(cache.residual[2:end], cache, u, p)

function Φ!(residual, cache::MIRKCache, u, p = cache.p)
    return Φ!(residual, cache.fᵢ_cache, cache.k_discrete, cache.f!, cache.TU,
        cache.y, u, p, cache.mesh, cache.mesh_dt, cache.stage)
end

@views function Φ!(residual, fᵢ_cache::DiffCache, k_discrete, f!, TU::MIRKTableau, y, u, p,
    mesh, mesh_dt, stage::Int)
    @unpack c, v, x, b = TU

    tmp = get_tmp(fᵢ_cache, u)
    T = eltype(u)
    for i in eachindex(k_discrete)
        K = get_tmp(k_discrete[i], u)
        residᵢ = get_tmp(residual[i], u)
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
