function BVPSystem{T}(fun::Function, bc::Function, x::Vector{T}, M::Integer, order)
    N = size(x,1)
    BVPSystem{T}(order, M, N, fun, bc, x, Matrix{T}(M,N), Matrix{T}(M,N), zeros(T,M,N))
end

# If user offers an intial guess.
function BVPSystem{T}(fun::Function, bc::Function, x::Vector{T}, y::Matrix{T}, order)
    M, N = size(y)
    BVPSystem{T}(order, M, N, fun, bc, x, y, Matrix{T}(M,N), zeros(T,M,N))
end

function BVPSystem{T,U}(fun::Function, bc::Function, x::Vector{T}, y::Matrix{U}, order)
    G = promote_type(T,U)
    BVPSystem(fun, bc, G.(x), G.(y), order)
end

# Auxiliary functions for evaluation
@inline function eval_fun!{T}(S::BVPSystem{T})
    for i in 1:S.N
        S.fun!(S.x[i], @view(S.y[:,i]), @view(S.f[:, i]))
    end
end

@inline eval_bc_residual!{T}(S::BVPSystem{T}) = S.bc!(@view(S.residual[:,end]), @view(S.y[:, 1]),
                                                    @view(S.y[:, end]))

function Φ!{T}(S::BVPSystem{T})
    M, N, residual, x, y, fun!, order = S.M, S.N, S.residual, S.x, S.y, S.fun!, S.order
    TU = constructMIRK(S)
    c, v, b, X, K = TU.c, TU.v, TU.b, TU.x, TU.K
    for i in 1:N-1
        h = x[i+1] - x[i]
        # Update K
        for r in 1:order
            x_new::T = x[i] + c[r]*h
            y_new::Vector{T} = (one(T)-v[r])*@view(y[:, i]) + v[r]*@view(y[:, i+1])
            if r > 1
                y_new += h * sum(j->X[r,j]*@view(K[:, j]), 1:r-1)
            end
            fun!(x_new, y_new, @view(K[:, r]))
        end
        # Update residual
        residual[:, i] = @view(y[:, i+1]) - @view(y[:, i]) - h * sum(j->b[j]*@view(K[:, j]), 1:order)
    end
    eval_bc_residual!(S)
end

# The whole MIRK scheme
function MIRK_scheme{T}(S::BVPSystem{T})
    # Upper-level iteration
    function nleq(z)
        z = reshape(z, S.M, S.N)
        copy!(S.y, z)
        Φ!(S)
        S.residual
    end
    NLsolve.nlsolve(NLsolve.not_in_place(nleq), vec(S.y))
    # Lower-levle iteration
    # continuousSolution = CMIRK(S.y)
    # eval_fun!(S)
    # residual = norm(D(continuousSolution)(S.y) - S.f)
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
