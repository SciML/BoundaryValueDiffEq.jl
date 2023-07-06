# Dispatches on BVPSystem
function BVPSystem(fun, bc, p, x, M::Integer, alg::Union{GeneralMIRK, MIRK})
    T = eltype(x)
    N = size(x, 1)
    y = vector_alloc(T, M, N)
    order = alg_order(alg)
    s = alg_stage(alg)
    BVPSystem(order, M, N, fun, bc, p, s, x, y, vector_alloc(T, M, N),
              vector_alloc(T, M, N),
              eltype(y)(undef, M))
end

# If user offers an intial guess
function BVPSystem(fun, bc, p, x, y, alg::Union{GeneralMIRK, MIRK})
    T, U = eltype(x), eltype(y)
    M, N = size(y)
    order = alg_order(alg)
    s = alg_stage(alg)
    BVPSystem{T, U}(order, M, N, fun, bc, p, s, x, y, vector_alloc(T, M, N),
                    vector_alloc(T, M, N), eltype(y)(M))
end

# Dispatch aware of eltype(x) != eltype(prob.u0)
function BVPSystem(prob::BVProblem, x, alg::Union{GeneralMIRK, MIRK})
    y = vector_alloc(prob.u0, x)
    M = length(y[1])
    N = size(x, 1)
    order = alg_order(alg)
    s = alg_stage(alg)
    BVPSystem(order, M, N, prob.f, prob.bc, prob.p, s, x, y, deepcopy(y),
              deepcopy(y), typeof(x)(undef, M))
end

# Auxiliary functions for evaluation
@inline function eval_fun!(S::BVPSystem)
    for i in 1:(S.N)
        S.fun!(S.f[i], S.y[i], S.p, S.x[i])
    end
end

@inline general_eval_bc_residual!(S::BVPSystem) = S.bc!(S.residual[end], S.y, S.p, S.x)
@inline function eval_bc_residual!(S::BVPSystem)
    S.bc!(S.residual[end], (S.y[1], S.y[end]), S.p, (S.x[1], S.x[end]))
end

function Φ!(S::BVPSystem{T}, TU::MIRKTableau, cache::AbstractMIRKCache) where {T}
    M, N, residual, x, y, fun!, s = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.s
    c, v, X, b = TU.c, TU.v, TU.x, TU.b
    K = similar([zeros(Float64, S.M)], S.s)
    for i in 1:(N - 1)
        h = x[i + 1] - x[i]
        # Update K
        for r in 1:s
            x_new = x[i] + c[r] * h
            y_new = (1 - v[r]) * y[i] + v[r] * y[i + 1]
            if r > 1
                y_new += h * sum(j -> X[r, j] * K[j], 1:(r-1))
            end
            temp = zeros(Float64, M)
            fun!(temp, y_new, S.p, x_new)
            K[r] = temp[:]
        end
        # Update residual
        residual[i] = y[i + 1] - y[i] - h * sum(j -> b[j] * K[j], 1:s)
        cache.k_discrete[i, :] = K[:]
    end
end
