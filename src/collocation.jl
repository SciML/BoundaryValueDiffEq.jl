# Dispatches on BVPSystem
function BVPSystem{T}(fun, bc, x::Vector{T}, M::Integer, order)
    N = size(x,1)
    y = vector_alloc(T, M, N)
    BVPSystem(order, M, N, fun, bc, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N))
end

# If user offers an intial guess
function BVPSystem{T,U<:AbstractArray}(fun, bc, x::Vector{T}, y::Vector{U}, order)
    M, N = size(y)
    BVPSystem{T,U}(order, M, N, fun, bc, x, y, vector_alloc(T, M, N), vector_alloc(T, M, N))
end

# Auxiliary functions for evaluation
@inline function eval_fun!{T}(S::BVPSystem{T})
    for i in 1:S.N
        S.fun!(S.x[i], S.y[i], S.f[i])
    end
end

@inline eval_bc_residual!{T}(S::BVPSystem{T}) = S.bc!(S.residual[end], S.y)

function Φ!{T}(S::BVPSystem{T}, TU::MIRKTableau)
    M, N, residual, x, y, fun!, order = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order
    c, v, b, X, K, D = TU.c, TU.v, TU.b, TU.x, TU.K, TU.D
    for i in 1:N-1
        h = x[i+1] - x[i]
        # Update K
        for r in 1:order
            x_new = x[i] + c[r]*h
            y_new = (one(T)-v[r])*y[i] + v[r]*y[i+1]
            if r > 1
                inc = zero(T)
                for j in 1:r-1
                    inc += X[r, j] * K[j]
                end
                y_new += h * inc
            end
            fun!(x_new, y_new, K[r])
            fun_jac!(D[r], fun!, x_new, y_new, K[r])
        end
        # Update residual
        slope = zero(T)
        for j in 1:order
            slope += b[j]*K[j]
        end
        residual[i] = y[i+1] - y[i] - h * slope
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
