# Dispatches on BVPSystem
function BVPSystem(fun, bc, p, x, M::Integer, order)
    T = eltype(x)
    N = size(x, 1)
    y = vector_alloc(T, M, N)
    BVPSystem(order, M, N, fun, bc, p, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N),
              eltype(y)(undef, M))
end

# If user offers an intial guess
function BVPSystem(fun, bc, p, x, y, order)
    T, U = eltype(x), eltype(y)
    M, N = size(y)
    BVPSystem{T, U}(order, M, N, fun, bc, p, x, y, vector_alloc(T, M, N),
                    vector_alloc(T, M, N), eltype(y)(M))
end

# Dispatch aware of eltype(x) != eltype(prob.u0)
function BVPSystem(prob::BVProblem, x, order)
    y = vector_alloc(prob.u0, x)
    M = length(y[1])
    N = size(x, 1)
    BVPSystem(order, M, N, prob.f, prob.bc, prob.p, x, y, deepcopy(y),
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
    M, N, residual, x, y, fun!, order = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order
    K, b = cache.K, TU.b
    c, v, X = TU.c, TU.v, TU.x
    for i in 1:(N - 1)
        h = x[i + 1] - x[i]
        # Update K
        for r in 1:order
            x_new = x[i] + c[r] * h
            y_new = (1 - v[r]) * y[i] + v[r] * y[i + 1]
            if r > 1
                y_new += h * sum(j -> X[r, j] * K[j], 1:(r - 1))
            end
            fun!(K[r], y_new, S.p, x_new)
        end
        # Update residual
        residual[i] = y[i + 1] - y[i] - h * sum(j -> b[j] * K[j], 1:order)
    end
end
