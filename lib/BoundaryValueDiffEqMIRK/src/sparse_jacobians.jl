# For MIRK Methods
"""
    __generate_sparse_jacobian_prototype(::MIRKCache, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, _, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::MIRKCache, ::TwoPointBVProblem, ya, yb, M, N)

Generate a prototype of the sparse Jacobian matrix for the BVP problem.

If the problem is a TwoPointBVProblem, then this is the complete Jacobian, else it only
computes the sparse part excluding the contributions from the boundary conditions.
"""
function __generate_sparse_jacobian_prototype(cache::MIRKCache, ya, yb, M, N)
    return __generate_sparse_jacobian_prototype(cache, cache.problem_type, ya, yb, M, N)
end

function __generate_sparse_jacobian_prototype(
        ::MIRKCache, ::StandardBVProblem, ya, yb, M, N
    )
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))
    return J_c
end

function __generate_sparse_jacobian_prototype(
        cache::MIRKCache, ::TwoPointBVProblem, ya, yb, M, N
    )
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # A BandedMatrix retains its bandwidth when resized, so a trust-region
    # solver cannot append a damping block below it. CSC permits that block
    # and underdetermined QR. It also supports LU of the symmetric normal
    # equations, for which Symmetric{BandedMatrix} has no LU implementation.
    # Other solvers retain banded storage.
    damping = cache.alg.optimize === nothing &&
        __needs_sparse_damping(cache.alg.nlsolve, cache.y₀_flat)
    return J₁ < J₂ || damping ? SparseArrays.sparse(J) : J
end

# Structural discovery and coloring run on the host before solves and after
# mesh refinement. Only
# index metadata and sparse storage are transferred to the device; neither the
# current state nor a numerically evaluated Jacobian is copied to the host.

function __mirk_device_boundary_pattern(
        prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size
    )
    M, nodes = size(y)
    nbc, nunknowns = prod(bc_sizes[1]), length(y)
    return __device_boundary_pattern(alg.jac_alg.bc_diffmode, eltype(y), nbc, nunknowns) do
        host_p = __device_host_parameter(p)
        mesh_dt = diff(host_mesh)
        iip = Val(isinplace(prob))
        function boundary!(residual, x)
            states = reshape(x, M, nodes)
            k = Array{eltype(x)}(undef, M, TU.s, nodes - 1)
            ki = Array{eltype(x)}(undef, M, ITU.s_star - TU.s, nodes - 1)
            for interval in 1:(nodes - 1)
                # Every stage may couple every component of the two adjacent
                # nodes. Union their dependencies without evaluating the RHS:
                # a GPU-compatible RHS need not be callable on host tracers.
                dependencies = sum(view(states, :, interval:(interval + 1)))
                fill!(view(k, :, :, interval), dependencies)
                fill!(view(ki, :, :, interval), dependencies)
            end
            sol = EvalSol(
                __build_interpolation(
                    states, k, ki, host_mesh, mesh_dt, Val(nameof(typeof(alg))), in_size
                )
            )
            __device_eval!(
                __device_reshape(residual, bc_sizes[1]), prob.f.bc,
                (
                    sol, get(prob.kwargs, :tune_parameters, false) ?
                        view(states, (M - length(host_p) + 1):M, 1) : host_p, host_mesh,
                ), iip
            )
            return nothing
        end
        return boundary!
    end
end

function __generate_sparse_jacobian_prototype(
        prob::BVProblem, alg::AbstractMIRK, y::AbstractMatrix, host_mesh,
        TU, ITU, bc_sizes, p, in_size = (size(y, 1),)
    )
    boundary = () -> __mirk_device_boundary_pattern(prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size)
    return __device_sparse_structure(prob.problem_type, alg.jac_alg, y, bc_sizes, 0, boundary)
end

function __mirk_prepare_device_jacobian(prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size)
    return __prepare_device_jacobian(y, prob.problem_type, bc_sizes) do
        __generate_sparse_jacobian_prototype(prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size)
    end
end
