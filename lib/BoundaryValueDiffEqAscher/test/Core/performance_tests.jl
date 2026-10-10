using BoundaryValueDiffEqAscher, SciMLBase, LinearAlgebra, SparseArrays, ForwardDiff, Test

const Ascher = BoundaryValueDiffEqAscher

function jacobian_test_problem(inplace, twopoint; matrix_state = false, ode = false)
    f(u, p, t) = reshape([u[2], -p * u[3], ode ? -u[3] + exp(u[1]) : u[3] - exp(u[1])], size(u))
    f!(du, u, p, t) = (du .= f(u, p, t); nothing)
    bca(u, p) = ode ? [u[1] + 0.2u[3]^2, u[2]] : [u[1] + 0.2u[3]^2]
    bcb(u, p) = [u[2] * u[3]]
    bca!(res, u, p) = (res .= bca(u, p); nothing)
    bcb!(res, u, p) = (res .= bcb(u, p); nothing)
    # Boundary dependencies change after the initial AD preparation. Both
    # branches also inspect algebraic variables and several interior points.
    function bc(u, p, t)
        point = u(0.173)[1] > 0 ? 0.811 : 0.419
        res = [u(point)[1] + u(0.619)[3]^2, u(0.287)[2] * u(0.903)[3]]
        ode && push!(res, u(0.0)[2])
        return res
    end
    bc!(res, u, p, t) = (res .= bc(u, p, t); nothing)
    u0 = matrix_state ? zeros(1, 3) : zeros(3)
    nleft = ode ? 2 : 1
    mass_matrix = ode ? I : Diagonal([1.0, 1.0, 0.0])
    if twopoint
        fun = ODEFunction(inplace ? f! : f; mass_matrix)
        return TwoPointBVProblem(
            fun, inplace ? (bca!, bcb!) : (bca, bcb), u0, (0.0, 1.0), 2.0;
            bcresid_prototype = (zeros(nleft), zeros(1)), nlls = Val(false)
        )
    end
    fun = BVPFunction(
        inplace ? f! : f, inplace ? bc! : bc;
        mass_matrix, bcresid_prototype = zeros(ode ? 3 : 2)
    )
    return BVProblem(fun, u0, (0.0, 1.0), 2.0; nlls = Val(false))
end

# Independent reference using the public interpolation representation, instead
# of the direct stage evaluation used by the optimized residual.
function reference_collocation(cache, x, p)
    sol = Ascher.__ascher_global_solution(cache, x)
    (; ncomp, ny, mesh, mesh_dt, k, TU) = cache
    result = eltype(x)[]
    weights = zeros(eltype(mesh), k)
    Ascher.rkbas!(1.0, TU.coef, k, weights)
    for i in 1:(length(mesh) - 1)
        for j in 1:k
            t = mesh[i] + mesh_dt[i] * TU.rho[j]
            u = sol(t)
            rhs = if isinplace(cache.prob)
                du = similar(u)
                cache.prob.f(du, u, p, t)
                vec(du)
            else
                vec(cache.prob.f(u, p, t))
            end
            append!(result, sol.stages[1:ncomp, j, i] - rhs[1:ncomp])
            append!(result, rhs[(ncomp + 1):(ncomp + ny)])
        end
        append!(result, sol.z[:, i + 1] - sol.z[:, i] - mesh_dt[i] * sol.stages[1:ncomp, :, i] * weights)
    end
    return result
end

@testset "Sparse collocation and dense reference" begin
    for inplace in (false, true), twopoint in (false, true), matrix_state in (false, true), ode in (false, true)
        prob = jacobian_test_problem(inplace, twopoint; matrix_state, ode)
        for alg in (Ascher1(), Ascher4(), Ascher7())
            cache = init(prob, alg; dt = 0.25, adaptive = false)
            nlprob = Ascher.__construct_nlproblem(cache)
            @test nlprob.f.jac_prototype isa SparseMatrixCSC
            for sign in (-1, 1)
                x = sign .* collect(range(0.1, 0.9; length = length(nlprob.u0)))
                loss = x -> begin
                    res = similar(x)
                    Ascher.__ascher_global_loss!(res, x, prob.p, cache)
                    res
                end
                Jreference = ForwardDiff.jacobian(loss, x)
                J = if inplace
                    J = copy(nlprob.f.jac_prototype)
                    nlprob.f.jac(J, x, prob.p)
                    J
                else
                    nlprob.f.jac(x, prob.p)
                end
                @test Matrix(J) ≈ Jreference rtol = 1.0e-10 atol = 1.0e-10
                if inplace
                    # The default nonlinear polyalgorithm can request a dense
                    # workspace when falling back to a trust-region solver.
                    Jdense = fill(NaN, size(Jreference))
                    nlprob.f.jac(Jdense, x, prob.p)
                    @test Jdense ≈ Jreference rtol = 1.0e-10 atol = 1.0e-10
                end
                ncoll = length(x) - (ode ? 3 : 2)
                res = zeros(ncoll)
                Ascher.__ascher_collocation_loss!(res, x, prob.p, cache)
                @test res ≈ reference_collocation(cache, x, prob.p) rtol = 1.0e-8 atol = 1.0e-8
            end
        end
    end
end

@testset "Sparse storage grows with the mesh" begin
    for twopoint in (false, true), ode in (false, true)
        prob = jacobian_test_problem(true, twopoint; ode)
        prototypes = map((0.1, 0.05)) do dt
            cache = init(prob, Ascher4(); dt, adaptive = false)
            Ascher.__construct_nlproblem(cache).f.jac_prototype
        end
        @test nnz(prototypes[2]) < 2.1nnz(prototypes[1])
        @test nnz(prototypes[2]) < length(prototypes[2]) / 4
    end
end

@testset "Explicit dense and finite-difference backends" begin
    function f!(du, u, p, t)
        du[1] = u[3]
        du[2] = -u[1]
        du[3] = u[2] + u[3]
        return nothing
    end
    function bc!(res, u, p, t)
        res[1] = u(0.0)[1] - 1
        res[2] = u(1.0)[2]
        return nothing
    end
    bca!(res, u, p) = (res[1] = u[1] - 1; nothing)
    bcb!(res, u, p) = (res[1] = u[2]; nothing)
    mass_matrix = Diagonal([1.0, 1.0, 0.0])
    for prob in (
            BVProblem(
                BVPFunction(f!, bc!; mass_matrix, bcresid_prototype = zeros(2)),
                zeros(3), (0.0, 1.0); nlls = Val(false)
            ),
            TwoPointBVProblem(
                ODEFunction(f!; mass_matrix), (bca!, bcb!),
                zeros(3), (0.0, 1.0); bcresid_prototype = (zeros(1), zeros(1)), nlls = Val(false)
            ),
        )
        reference = solve(prob, Ascher4(); dt = 0.1, adaptive = false, abstol = 1.0e-10)
        for mode in (AutoForwardDiff(), AutoFiniteDiff(), AutoSparse(AutoFiniteDiff()))
            alg = Ascher4(; jac_alg = BVPJacobianAlgorithm(mode))
            cache = init(prob, alg; dt = 0.1, adaptive = false)
            @test cache.alg.jac_alg.diffmode === mode
            prototype = Ascher.__construct_nlproblem(cache).f.jac_prototype
            @test (prototype isa SparseMatrixCSC) == (mode isa AutoSparse)
            sol = solve(prob, alg; dt = 0.1, adaptive = false, abstol = 1.0e-10)
            @test successful_retcode(sol)
            @test sol(0.371) ≈ reference(0.371) atol = 1.0e-8
        end
    end
end

@testset "BVP sparse solve and adaptive interpolation" begin
    f(u, p, t) = [u[2], -u[1]]
    f!(du, u, p, t) = (du .= f(u, p, t); nothing)
    bca(u, p) = [u[1]]
    bcb(u, p) = [u[1] - 1]
    bca!(res, u, p) = (res .= bca(u, p); nothing)
    bcb!(res, u, p) = (res .= bcb(u, p); nothing)
    bc(u, p, t) = [u(0.0)[1], u(pi / 2)[1] - 1]
    bc!(res, u, p, t) = (res .= bc(u, p, t); nothing)
    for inplace in (false, true), twopoint in (false, true), adaptive in (false, true)
        prob = if twopoint
            TwoPointBVProblem(
                inplace ? f! : f, inplace ? (bca!, bcb!) : (bca, bcb),
                [0.1, 0.1], (0.0, pi / 2); bcresid_prototype = (zeros(1), zeros(1))
            )
        else
            BVProblem(inplace ? f! : f, inplace ? bc! : bc, [0.1, 0.1], (0.0, pi / 2))
        end
        for alg in (Ascher2(), Ascher4(), Ascher7())
            cache = init(prob, alg; dt = 0.1, adaptive)
            @test Ascher.__construct_nlproblem(cache).f.jac_prototype isa SparseMatrixCSC
            sol = solve!(cache)
            @test successful_retcode(sol)
            for t in (0.0, 0.371, pi / 2)
                @test sol(t) ≈ [sin(t), cos(t)] atol = 2.0e-5
            end
        end
    end
end
