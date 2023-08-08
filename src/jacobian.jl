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

## Support only ForwardDiff and FiniteDiff here. All others via extensions!

### Setup Caches
function _mirk_jacobian_setup end
function _mirk_complete_jacobian_setup end
function _mirk_dense_jacobian_setup end
function _mirk_sparse_jacobian_setup end

function _mirk_sparse_jacobian_setup(::AutoSparseFiniteDiff, S::BVPSystem, y, p)
    J₀ = BandedMatrix(similar(y, (S.M * S.N, S.M * S.N)), (S.M + 1, S.M))[(S.M + 1):end, :]
    return FiniteDiff.JacobianCache(similar(y), similar(y)[(S.M + 1):end];
        colorvec = matrix_colors(J0), sparsity = sparse(J0))
end

### Compute the Jacobian
function __mirk_compute_jacobian end
function __mirk end
function __mirk_compute_sparse_jacobian! end
## We can potentially do sparsity detection on the boundary conditions as well. But for now
## we will just use the dense version.
function __mirk_compute_dense_jacobian! end
