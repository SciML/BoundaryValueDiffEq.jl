# Dispatches on BVPSystem
function BVPSystem(fun, bc, p, x::Vector{T}, M::Integer, order) where T
    N = size(x,1)
    y = vector_alloc(T, M, N)
    BVPSystem(order, M, N, fun, bc, p, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N), eltype(y)(undef, M))
end

# If user offers an intial guess
function BVPSystem(fun, bc, p, x::Vector{T}, y::Vector{U}, order) where {T,U<:AbstractArray}
    M, N = size(y)
    BVPSystem{T,U}(order, M, N, fun, bc, p, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N), eltype(y)(M))
end

# Auxiliary functions for evaluation
@inline function eval_fun!(S::BVPSystem)
    for i in 1:S.N
        S.fun!(S.f[i], S.y[i], S.p, S.x[i])
    end
end

@inline general_eval_bc_residual!(S::BVPSystem) = S.bc!(S.residual[end], S.y, S.p, S.x)
@inline eval_bc_residual!(S::BVPSystem) = S.bc!(S.residual[end], (S.y[1], S.y[end]), S.p, (S.x[1], S.x[end]))

#=
@inline function banded_update_K!(S::BVPSystem, cache::AbstractMIRKCache, TU::MIRKTableau, i, h)
    M, N, residual, x, y, fun!, order, y_new = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order, S.tmp
    c, v, b, X = TU.c, TU.v, TU.b, TU.x
    ## K, LJ, RJ, Jacobian = cache.K, cache.LJ, cache.RJ, cache.Jacobian
    K = cache.K

    ## index = (M*(i-1)+2):(M*i+1)
    ## Lindex = (M*(i-1)+1):(M*i)
    ## Rindex = (M*(i-1)+1+M):(M*i+M)

    function Kᵣ!(Kr, y, y₁, r)
        x_new = x[i] + c[r]*h
        y_new = (1-v[r])*y + v[r]*y₁
        if r > 1
          y_new += h * sum(j->X[r, j]*K[j], 1:r-1)
        end
        fun!(x_new, y_new, Kr)
    end

    # L is the left strip, and R is the right strip
    # Lᵢ = -I - hᵢ*Σᵣbᵣ*(∂Kᵣ/∂yᵢ)
    # Rᵢ = I - hᵢ*Σᵣbᵣ*(∂Kᵣ/∂y_{i+1})
    # From the paper "A Runge-Kutta Type Boundary Value ODE Solver with Defect Control"
    # by W.H. Enright and Paul Muir
    for r in 1:order
        Kᵣ!(K[r], y[i], y[i+1], r)
        # ∂Kᵣ/∂yᵢ
        ### ForwardDiff.jacobian!(LJ[r], (Kr, y₀)->Kᵣ!(Kr, y₀, y[i+1], r), K[r], y[i])
        # ∂Kᵣ/∂y_{i+1}
        ## ForwardDiff.jacobian!(RJ[r], (Kr, y₁)->Kᵣ!(Kr, y[i], y₁, r), K[r], y[i+1])
        # h*bᵣ*(∂Kᵣ/∂yᵢ)
        ## scale!(-b[r]*h, LJ[r])
        # hᵢ*Σᵣbᵣ*(∂Kᵣ/∂y_{i+1})
        ## scale!(-b[r]*h, RJ[r])
        # sum them up
        ## Jacobian[index,Lindex] += LJ[r]
        ## Jacobian[index,Rindex] += RJ[r]
        # fun_jac!(LJ[r], fun!, x_new, y_new, K[r])
        # fun_jac!(RJ[r], fun!, x_new, y_new, K[r])
    end
    # Lᵢ = -I - ...
    # Rᵢ = I - ...
    ## Jacobian[index,Lindex] -= I
    ## Jacobian[index,Rindex] += I
end

function banded_Φ!(S::BVPSystem, TU::MIRKTableau, cache::AbstractMIRKCache)
    order, residual, N, y, x = S.order, S.residual, S.N, S.y, S.x
    K, b = cache.K, TU.b
    for i in 1:N-1
        h = x[i+1] - x[i]
        banded_update_K!(S, cache, TU, i, h)
        # Update residual
        residual[i] = y[i+1] - y[i] - h * sum(j->b[j]*K[j], 1:order)
    end
    eval_bc_residual!(S)
    #ForwardDiff.jacobian!(@view(cache.Jacobian[1:S.M, 1:S.M]),                     (x,y)->S.bc!(x,y,S.y[1]),   residual[1],   S.y[1])
    #ForwardDiff.jacobian!(@view(cache.Jacobian[(end-S.M+1):end, (end-S.M+1):end]), (x,y)->S.bc!(x,S.y[end],y), residual[end], S.y[end])
    #display(cache.Jacobian)
end
=#

@inline function update_K!(S::BVPSystem, cache::AbstractMIRKCache, TU::MIRKTableau, i, h)
    M, N, residual, x, y, fun!, order = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order
    K, b = cache.K, TU.b
    c, v, X = TU.c, TU.v, TU.x

    function Kᵣ!(Kr, y, y₁, r)
        x_new = x[i] + c[r]*h
        y_new = (1-v[r])*y + v[r]*y₁
        if r > 1
          y_new += h * sum(j->X[r, j]*K[j], 1:r-1)
        end
        fun!(Kr, y_new, S.p, x_new)
    end

    for r in 1:order
        Kᵣ!(K[r], y[i], y[i+1], r)
    end
end

function Φ!(S::BVPSystem{T}, TU::MIRKTableau, cache::AbstractMIRKCache) where T
    M, N, residual, x, y, fun!, order = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order
    K, b = cache.K, TU.b
    c, v, X = TU.c, TU.v, TU.x
    for i in 1:N-1
        h = x[i+1] - x[i]
        # Update K
        update_K!(S, cache, TU, i, h)
        # Update residual
        residual[i] = y[i+1] - y[i] - h * sum(j->b[j]*K[j], 1:order)
    end
end
