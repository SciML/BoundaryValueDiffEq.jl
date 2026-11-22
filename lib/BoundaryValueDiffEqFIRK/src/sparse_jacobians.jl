# For FIRK Methods
"""
    __generate_sparse_jacobian_prototype(::FIRKCacheNested, ::StandardBVProblem, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::FIRKCacheNested, ::TwoPointBVProblem, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::FIRKCacheExpand, ::StandardBVProblem, ya, yb, M, N)
    __generate_sparse_jacobian_prototype(::FIRKCacheExpand, ::TwoPointBVProblem, ya, yb, M, N)

Generate a prototype of the sparse Jacobian matrix for the BVP problem.

If the problem is a TwoPointBVProblem, then this is the complete Jacobian, else it only
computes the sparse part excluding the contributions from the boundary conditions.
"""
function __generate_sparse_jacobian_prototype(
        ::FIRKCacheNested, ::StandardBVProblem, ya, yb, M, N
    )
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J_c = BandedMatrix(Ones{eltype(ya)}(M * (N - 1), M * N), (1, 2M - 1))
    return J_c
end

function __generate_sparse_jacobian_prototype(
        cache::FIRKCacheNested, ::TwoPointBVProblem, ya, yb, M, N
    )
    fast_scalar_indexing(ya) ||
        error("Sparse Jacobians are only supported for Fast Scalar Index-able Arrays")
    J₁ = length(ya) + length(yb) + M * (N - 1)
    J₂ = M * N
    J = BandedMatrix(Ones{eltype(ya)}(J₁, J₂), (M + 1, M + 1))
    # Trust-region damping appends rows outside the original band. Sparse storage
    # also supports the QR needed by underdetermined systems.
    damping = cache.alg.optimize === nothing &&
        __needs_sparse_damping(cache.alg.nlsolve, ya)
    return J₁ < J₂ || damping ? sparse(J) : J
end

function __generate_sparse_jacobian_prototype(
        cache::FIRKCacheExpand, ::StandardBVProblem, ya, yb, M, N
    )
    (; stage) = cache

    # Get number of nonzeros
    block_size = M * (stage + 1) * M * (stage + 2)
    l = (N - 1) * block_size
    # Initialize Is and Js
    Is = Vector{Int}(undef, l)
    Js = Vector{Int}(undef, l)

    # Fill Is and Js
    row_size = M * (stage + 1) * (N - 1)

    idx = 1
    i_start = 0
    j_start = 0
    i_step = M * (stage + 1)
    j_step = M * (stage + 2)
    for k in 1:(N - 1)
        for i in 1:i_step
            for j in 1:j_step
                Is[idx] = i + i_start
                Js[idx] = j + j_start
                idx += 1
            end
        end
        i_start += i_step
        j_start += i_step
    end

    # Create sparse matrix from Is and Js
    J_c = _sparse_like(Is, Js, ya, row_size, row_size + M)
    return J_c
end

function __generate_sparse_jacobian_prototype(
        cache::FIRKCacheExpand, ::TwoPointBVProblem, ya, yb, M, N
    )
    (; stage) = cache

    # Get number of nonzeros
    block_size = M * (stage + 1) * M * (stage + 2)
    l = (N - 1) * block_size + M * (stage + 2) * (length(ya) + length(yb))
    # Initialize Is and Js
    Is = Vector{Int}(undef, l)
    Js = Vector{Int}(undef, l)

    # Fill Is and Js
    row_size = M * (stage + 1) * (N - 1)
    idx = 1
    i_start = 0
    j_start = 0
    i_step = M * (stage + 1)
    j_step = M * (stage + 2)

    # Fill first rows
    for i in 1:length(ya)
        for j in 1:j_step
            Is[idx] = i
            Js[idx] = j
            idx += 1
        end
    end
    i_start += length(ya)

    for k in 1:(N - 1)
        for i in 1:i_step
            for j in 1:j_step
                Is[idx] = i + i_start
                Js[idx] = j + j_start
                idx += 1
            end
        end
        i_start += i_step
        j_start += i_step
    end
    j_start -= i_step
    #Fill last rows
    for i in 1:length(yb)
        for j in 1:j_step
            Is[idx] = i + i_start
            Js[idx] = j + j_start
            idx += 1
        end
    end

    # Create sparse matrix from Is and Js
    J = _sparse_like(Is, Js, ya, row_size + length(ya) + length(yb), row_size + M)

    return J
end

# Structural discovery and coloring run on the host before solves and after
# mesh refinement. Only
# index metadata and sparse storage are transferred to the device; neither the
# current state nor a numerically evaluated Jacobian is copied to the host.

function __firk_device_boundary_pattern(
        prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size
    )
    M, nodes = size(y)
    stencil_stage = alg.nested_nlsolve ? 0 : TU.s
    nbc, nunknowns = prod(bc_sizes[1]), length(y)
    return __device_boundary_pattern(alg.jac_alg.bc_diffmode, eltype(y), nbc, nunknowns) do
        host_p = __device_host_parameter(p)
        mesh_dt = diff(host_mesh)
        iip = Val(isinplace(prob))
        function boundary!(residual, x)
            states = reshape(x, M, nodes)
            coefficients = Array{eltype(x)}(undef, M, 6, length(host_mesh) - 1)
            for i in 1:(length(host_mesh) - 1)
                ctr = (i - 1) * (stencil_stage + 1) + 1
                dependencies = sum(view(states, :, ctr:(ctr + stencil_stage + 1)))
                fill!(view(coefficients, :, :, i), dependencies)
            end
            sol = __firk_eval_sol(states, host_mesh, mesh_dt, coefficients, in_size, stencil_stage)
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
        prob::BVProblem, alg::AbstractFIRK, y::AbstractMatrix, host_mesh,
        TU, ITU, bc_sizes, p, in_size = (size(y, 1),)
    )
    boundary = () -> __firk_device_boundary_pattern(prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size)
    return __device_sparse_structure(prob.problem_type, alg.jac_alg, y, bc_sizes, alg.nested_nlsolve ? 0 : TU.s, boundary)
end

function __firk_prepare_device_jacobian(prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size)
    return __prepare_device_jacobian(y, prob.problem_type, bc_sizes) do
        __generate_sparse_jacobian_prototype(prob, alg, y, host_mesh, TU, ITU, bc_sizes, p, in_size)
    end
end
