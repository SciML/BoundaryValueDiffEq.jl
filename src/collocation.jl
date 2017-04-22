@inline function eval_fun!(f_out, fun!, x, y, p)
    for i in 1:size(f_out,1)
        fun!(view(f_out, i, :), x[i], view(y,i,:), p)
    end
end

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

function collocation_points!(f, y_m, f_m, residual, fun!, x, y, p, h)
    eval_fun!(f, fun!, x, y, p)
    eval_y_middle!(y_m, f, y, h)
    eval_fun!(f_m, fun!, x[1:end-1] + 1//2 * h, y_m, p)
    eval_col_residual!(residual, f, y, h, f_m)
end

function allocate_arrays(n, m)
    f = Array{Float64}(m,n)
    y_m = Array{Float64}(m-1,n)
    f_m = Array{Float64}(m-1,n)
    residual = Array{Float64}(m-1, n)
    f, y_m, f_m, residual
end

test_col() = begin
    fun!(out, x, y, p) = begin
       out[1] = p[2]*x*y[2]
       out[2] = p[1]*x*-exp.(y[1])
    end
    n, m = 2,4
    x = collect(linspace(0,1,m))
    h = x[2]-x[1]
    y = zeros(m,n)
    p = ones(2)
    f, y_m, f_m, residual = allocate_arrays(n, m)
    collocation_points!(f, y_m, f_m, residual, fun!, x, y, p, h)
    n,m,x,y,h,y,p,f, y_m, f_m, residual
end
n,m,x,y,h,y,p,f, y_m, f_m, residual = test_col()
