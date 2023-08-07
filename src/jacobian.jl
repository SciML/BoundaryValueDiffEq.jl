struct BVPJacobianWrapper{LossType} <: Function
    loss::LossType
end
(jw::BVPJacobianWrapper)(resid, u, p) = jw.loss(resid, u, p)
(jw::BVPJacobianWrapper)(u, p) = (resid = similar(u); jw.loss(resid, u, p); resid)

# FIXME: This is a nightmarish way to do this. And I appologize to anyone who glazes upon
#        this horror.
function _construct_nonlinear_problem_with_jacobian(f!::BVPJacobianWrapper, S::BVPSystem,
    y, p)
    J0 = BandedMatrix(similar(y, (S.M * S.N, S.M * S.N)), (S.M + 1, S.M))[(S.M + 1):end, :]
    jac_cache = JacobianCache(similar(y), similar(y)[(S.M + 1):end];
        colorvec = matrix_colors(J0), sparsity = sparse(J0))
    function jac!(J, x, p)
        function _f!(F, x)
            FF = similar(x)
            f!(FF, x, p)
            F .= FF[(S.M + 1):end]
            return F
        end
        finite_difference_jacobian!(@view(J[(S.M + 1):end, :]), _f!, x, jac_cache)
        function _f1!(F, x)
            FF = similar(x)
            f!(FF, x, p)
            F .= FF[begin:(S.M)]
            return
        end
        finite_difference_jacobian!(@view(J[begin:(S.M), :]), _f1!, x)
        return J
    end
    return NonlinearProblem(NonlinearFunction{true}(f!; jac = jac!), y, p)
end
