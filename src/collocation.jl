# Dispatches on BVPSystem
function BVPSystem{T}(fun, bc, x::Vector{T}, M::Integer, order)
    N = size(x,1)
    y = vector_alloc(T, M, N)
    BVPSystem(order, M, N, fun, bc, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N), eltype(y)(M))
end

# If user offers an intial guess
function BVPSystem{T,U<:AbstractArray}(fun, bc, x::Vector{T}, y::Vector{U}, order)
    M, N = size(y)
    BVPSystem{T,U}(order, M, N, fun, bc, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N), eltype(y)(M))
end

# Auxiliary functions for evaluation
@inline function eval_fun!(S::BVPSystem)
    for i in 1:S.N
        S.fun!(S.x[i], S.y[i], S.f[i])
    end
end

@inline eval_bc_residual!(S::BVPSystem) = S.bc!(S.residual[end], S.y)

@inline function update_K!(S::BVPSystem, cache::MIRK4Cache, TU::MIRKTableau, i, h)
    M, N, residual, x, y, fun!, order, y_new = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order, S.tmp
    c, v, b, X = TU.c, TU.v, TU.b, TU.x
    K, LJ, RJ, Jacobian = cache.K, cache.LJ, cache.RJ, cache.Jacobian

    indexDiag = (M*(i-1)+1):M*i
    # Left strip of the Jacobian is `Jacobian[indexDiag, indexDiag]`
    # Right strip of the Jacobian is `Jacobian[indexDiag, indexDiag+M]`

    function Kᵣ!(Kr, y, y₁, r)
        x_new = x[i] + c[r]*h
        y_new = (1-v[r])*y + v[r]*y₁
        if r > 1
          y_new += h * sum(j->X[r, j]*K[j], 1:r-1)
        end
        fun!(x_new, y_new, Kr)
    end

    # jcg = ForwardDiff.JacobianConfig((Kr,y)->Kᵣ!(Kr, y, y[1], 1), K[1], y[i])

    # L is the left strip, and R is the right strip
    # Lᵢ = -I - hᵢ*Σᵣbᵣ*(∂Kᵣ/∂yᵢ)
    # Rᵢ = I - hᵢ*Σᵣbᵣ*(∂Kᵣ/∂y_{i+1})
    # From the paper "A Runge-Kutta Type Boundary Value ODE Solver with Defect Control"
    # by W.H. Enright and Paul Muir
    for r in 1:order
        Kᵣ!(K[r], y[i], y[i+1], r)
        # ∂Kᵣ/∂yᵢ
        ## ForwardDiff.jacobian!(LJ[r], (Kr, y₀)->Kᵣ!(Kr, y₀, y[i+1], r), K[r], y[i])
        # ∂Kᵣ/∂y_{i+1}
        ## ForwardDiff.jacobian!(RJ[r], (Kr, y₁)->Kᵣ!(Kr, y[i], y₁, r), K[r], y[i+1])
        # h*bᵣ*(∂Kᵣ/∂yᵢ)
        ## scale!(-b[r]*h, LJ[r])
        # hᵢ*Σᵣbᵣ*(∂Kᵣ/∂y_{i+1})
        ## scale!(-b[r]*h, RJ[r])
        # sum them up
        ## Jacobian[indexDiag, indexDiag] += LJ[r]
        ## Jacobian[indexDiag, indexDiag+M] += RJ[r]
    end
    # Lᵢ = -I - ...
    # Rᵢ = I - ...
    ## Jacobian[indexDiag, indexDiag] -= I
    ## Jacobian[indexDiag, indexDiag+M] += I
end

function Φ!(S::BVPSystem, TU::MIRKTableau, cache::AbstractMIRKCache)
    order, residual, N, y, x = S.order, S.residual, S.N, S.y, S.x
    K, b = cache.K, TU.b
    for i in 1:N-1
        h = x[i+1] - x[i]
        update_K!(S, cache, TU, i, h)
        # Update residual
        residual[i] = y[i+1] - y[i] - h * sum(j->b[j]*K[j], 1:order)
    end
    eval_bc_residual!(S)
end

#=
[1]: J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27, Number
3, pp. 299-316, 2001.

But this implementation may don't use it.
=#

#=
@inline function eval_y_middle!(y_m, f, y, h)
    for i in 1:size(f, 1)-1
        y_m[i, :] = 1//2 * (y[i+1, :] + y[i, :]) - 1//8 * h * (f[i+1, :] - f[i, :])
    end
end

@inline function eval_col_residual!(residual, f, y, h, f_m)
    for i in 1:size(f, 1)-1
        residual[i, :] = y[i+1, :] - y[i, :] - h * 1//6 * (f[i, :] + f[i+1, :] + 4 * f_m[i])
    end
end

function collocation_points!(f, y_m, f_m, residual, fun!, x, y, h)
    eval_fun!(f, fun!, x, y)
    eval_y_middle!(y_m, f, y, h)
    eval_fun!(f_m, fun!, x[1:end-1] + 1//2 * h, y_m)
    eval_col_residual!(residual, f, y, h, f_m)
end

# TODO: A better allocation method needs to be done
# i.e. what if user gives a initial evaluation of
# the system of ODE.
function allocate_arrays(T, n, m)
    f = Array{T}(m,n)
    y_m = Array{T}(m-1,n) #See [1]
    f_m = Array{T}(m-1,n)
    residual = Array{T}(m-1, n)
    f, y_m, f_m, residual
end

# Just a testing function for development

test_col() = begin
    fun!(out, x, y) = begin
       out[1] = x*y[2]
       out[2] = x*-exp.(y[1])
@inline function eval_fun!(f_out, fun!, x, y)
    for i in 1:size(f_out,1)
        fun!(view(f_out, i, :), x[i], view(y,i,:))
    end
end

    end
    n, m = 2,4
    x = collect(linspace(0,1,m))
    h = x[2]-x[1]
    y = zeros(m,n)
    f, y_m, f_m, residual = allocate_arrays(Float64, n, m)
    collocation_points!(f, y_m, f_m, residual, fun!, x, y, h)
    n,m,x,y,h,y,f, y_m, f_m, residual
end
n,m,x,y,h,y,f, y_m, f_m, residual = test_col()
=#
