using BoundaryValueDiffEqMIRK, ADTypes, FastAlmostBandedMatrices, Test
using SciMLBase: BVProblem, init, solve, successful_retcode

const MIRK = BoundaryValueDiffEqMIRK

@testset "Dense boundary storage with sparse differentiation" begin
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = 0; nothing)
    f(u, p, t) = [u[2], zero(u[1])]
    function bc!(r, u, p, t)
        a, b, c = u(t[1]), u(t[end]), u((t[1] + t[end]) / 2)
        r[1] = a[1]^2 - 1
        r[2] = b[1]^2 + c[1]^2 - 6.25
        return nothing
    end
    function bc(u, p, t)
        a, b, c = u(t[1]), u(t[end]), u((t[1] + t[end]) / 2)
        return [a[1]^2 - 1, b[1]^2 + c[1]^2 - 6.25]
    end

    for iip in (true, false), mode in (
                AutoSparse(AutoForwardDiff()), AutoSparse(AutoFiniteDiff()), AutoForwardDiff(),
            )
        prob = BVProblem(
            iip ? f! : f, iip ? bc! : bc, [1.2, 0.8], (0.0, 1.0);
            nlls = Val(false)
        )
        alg = MIRK4(;
            nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm(;
                bc_diffmode = mode, nonbc_diffmode = AutoSparse(AutoForwardDiff())
            )
        )
        cache = init(prob, alg; dt = 0.25, adaptive = false)
        copyto!(cache.y₀_flat, vec(cache.y₀))
        nl = MIRK.__construct_problem(cache, copy(cache.y₀_flat), copy(cache.y₀))
        @test nl.f.jac_prototype isa AlmostBandedMatrix
        @test fillpart(nl.f.jac_prototype) isa Matrix
        @test size(fillpart(nl.f.jac_prototype)) == (2, length(nl.u0))

        # Compare the entire assembled matrix against independent dense AD.
        # Zero derivatives followed by nonzero ones expose stale fill entries.
        for x in (copy(nl.u0), zero(nl.u0), 1.1 .* nl.u0)
            J = if iip
                J = copy(nl.f.jac_prototype)
                fill!(fillpart(J), NaN)
                nl.f.jac(J, x, nl.p)
                J
            else
                fill!(fillpart(nl.f.jac_prototype), NaN)
                nl.f.jac(x, nl.p)
            end
            reference = if iip
                MIRK.ForwardDiff.jacobian(x) do z
                    r = similar(z)
                    nl.f(r, z, nl.p)
                    r
                end
            else
                MIRK.ForwardDiff.jacobian(z -> nl.f(z, nl.p), x)
            end
            @test Matrix(J) ≈ reference atol = 1.0e-6 rtol = 1.0e-6
        end

        sol = solve(prob, alg; dt = 0.25, adaptive = false, abstol = 1.0e-9)
        @test successful_retcode(sol)
        @test maximum(abs(sol.u[i][1] - (1 + sol.t[i])) for i in eachindex(sol.t)) < 1.0e-6
        @test maximum(abs(u[2] - 1) for u in sol.u) < 1.0e-6
    end

    # The normal MIRK entry point must benefit without an explicit linear solver.
    prob = BVProblem(f!, bc!, [1.2, 0.8], (0.0, 1.0); nlls = Val(false))
    sol = solve(prob, MIRK4(); dt = 0.25, adaptive = false, abstol = 1.0e-9)
    @test successful_retcode(sol)
    @test maximum(abs(sol.u[i][1] - (1 + sol.t[i])) for i in eachindex(sol.t)) < 1.0e-6
end
