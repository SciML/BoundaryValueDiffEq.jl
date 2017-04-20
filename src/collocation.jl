fun!(out, x, y, p) = begin
   out[1] = p[2]*x*y[2]
   out[2] = p[1]*x*-exp.(y[1])
end

@inline function eval_fun!(f_out, fun!, x, y, p)
    for i in 1:size(f_out,1)
        fun!(view(f_out, i, :), x[i], view(y,i,:), p)
    end
end

# y_m = Array{T}(m-1, n)
@inline function eval_y_middle!(y_m, f, y, h)
    for i in 1:size(f, 1)-1
        y_m[i, :] = 1//2 * (y[i+1, :] + y[i, :]) - 1//8 * h * (f[i+1, :] - f[i, :])
    end
end

@inline function eval_col_residual!(residual, f, y, h, f_m)
    for i in 1:size(f, 1)-1
        residual[i] = y[i+1, :] - y[i, :] - h * 1//6 * (f[i, :] + f[:, i+1] + 4 * f_m[i])
    end
end

function collocation_points!(f, y_m, f_m, residual, fun, x, y, p, h)
    eval_fun!(f, fun, x, y, p)
    # copy!(y_m, 1//2 * (y[2:end, :] + y[1:end-1, :]) - 1//8 * h * (f[2:end, :] - f[1:end-1, :]))
    eval_y_middle!(y_m, f, y, h)
    eval_fun(f_m, x[1:end-1] + 1//2 * h, y_m, p)
    # copy!(residual, y[2:end, :] - y[1:end-1, :] - h * 1//6 * (f[1:end-1, :] + f[:, 2:end] + 4 * f_m))
    eval_col_residual!(residual, f, y, h, f_m)
end

function system_jac(n, m, k, i_jac, j_jac, df_dy, df_dy_m, df_dp, df_dp_m, dbc_da, dbc_db, dbc_dp)

end
