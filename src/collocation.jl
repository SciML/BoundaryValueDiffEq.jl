using NLsolve

# ODE BVP problem system
immutable BVPSystem{T}
    M::Integer          # Number of equations in the ODE system
    N::Integer          # Number of nodes in the mesh
    fun!::Function      # N -> N
    bc!::Function       # 2 -> 2
    x::Vector{T}        # M
    y::Matrix{T}        # M*N
    f::Matrix{T}        # M*N
    residual::Matrix{T} # M*N
end

function BVPSystem{T}(fun::Function, bc::Function, x::Vector{T}, M::Integer)
    N = size(x,1)
    BVPSystem{T}(M, N, fun, bc, x, Matrix{T}(M,N), Matrix{T}(M,N), zeros(T,M,N))
end

# If user offers an intial guess.
function BVPSystem{T}(fun::Function, bc::Function, x::Vector{T}, y::Matrix{T})
    M, N = size(y)
    BVPSystem{T}(M, N, fun, bc, x, y, Matrix{T}(M,N), zeros(T,M,N))
end

function BVPSystem{T,U}(fun::Function, bc::Function, x::Vector{T}, y::Matrix{U})
    G = promote_type(T,U)
    BVPSystem(fun, bc, G.(x), G.(y))
end

# Auxiliary functions for evaluation
@inline function eval_fun!(S::BVPSystem)
    for i in 1:S.N
        S.fun!(S.x[i], @view(S.y[:,i]), @view(S.f[:, i]))
    end
end

@inline eval_bc_residual!(S::BVPSystem) = S.bc!(S.residual, @view(S.y[:, 1]), @view(S.y[:, end]))

function Φ!(fun!, bc!, x, y, residual, M, N, order::Integer)
    TU = constructMIRK(Val{order}, M, y)
    # M, N, residual, x, y, fun! = S.M, S.N, S.residual, S.x, S.y, S.fun!
    c, v, b, X, K = TU.c, TU.v, TU.b, TU.x, TU.K
    for i in 2:N-1
        h = x[i+1] - x[i]
        residual[:, i] = -@view(y[:, i]) + @view(y[:, i+1])

        # Update K
        for r in 1:order
            x_new = x[i] + c[r]*h
            y_new = (1-v[r])*y[i] + v[r]*@view(y[:, i+1])
            if r > 1
                y_new += h * sum(j->X[j,r]*@view(K[:, j]), 1:r-1)
            end
            fun!(x_new, y_new, @view(K[:, r]))
        end

        residual[:, i] -= h * sum(j->b[j]*@view(K[:, j]), 1:order)
    end
    bc!(residual, @view(y[:, 1]), @view(y[:, end]))
end

# Testing function for development, please ignore.
function func!(x, y, out)
    out[1] = y[2]
    out[2] = exp(-y[1])
end

function boundary!(residual, ua, ub)
    residual[1, 1] = ua[1]
    residual[1, end] = ub[1]
end

S = BVPSystem(func!, boundary!, Float64.(collect(1:5)), 2);
# eval_fun!(S)
# eval_bc_residual!(S)

# The whole MIRK scheme
function MIRK_scheme(S::BVPSystem, order)
    # Upper-level iteration
    function nleq(z)
        z = reshape(z, S.M, S.N)
        Φ!(S.fun!, S.bc!, S.x, z, S.residual, S.M, S.N, order)
        S.residual
    end
    nlsolve(not_in_place(nleq), vec(S.y))
    # Lower-levle iteration
    # continuousSolution = CMIRK(S.y)
    # eval_fun!(S)
    # residual = norm(D(continuousSolution)(S.y) - S.f)
end

sol = MIRK_scheme(S)

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
