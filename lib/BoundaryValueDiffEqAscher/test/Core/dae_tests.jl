using BoundaryValueDiffEqAscher, SciMLBase, LinearAlgebra, ForwardDiff, Test

# Every fixture is in semi-explicit form, with the differential variables first.
# Keep the algebraic equations in the solve rather than eliminating them.
function dae_problem(fixture, inplace, twopoint)
    (; f, bca, bcb, u0, tspan, p, ndiff) = fixture
    mass_matrix = Diagonal([ones(ndiff); zeros(length(u0) - ndiff)])
    f!(du, u, p, t) = (du .= f(u, p, t); nothing)
    bca!(res, u, p) = (res .= bca(u, p); nothing)
    bcb!(res, u, p) = (res .= bcb(u, p); nothing)
    prototype = (zeros(length(bca(u0, p))), zeros(length(bcb(u0, p))))
    if twopoint
        fun = ODEFunction(inplace ? f! : f; mass_matrix)
        bc = inplace ? (bca!, bcb!) : (bca, bcb)
        return TwoPointBVProblem(
            fun, bc, copy(u0), tspan, p; bcresid_prototype = prototype, nlls = Val(false)
        )
    end
    bc(u, p, t) = [bca(u(first(tspan)), p); bcb(u(last(tspan)), p)]
    bc!(res, u, p, t) = (res .= bc(u, p, t); nothing)
    fun = BVPFunction(
        inplace ? f! : f, inplace ? bc! : bc;
        mass_matrix, bcresid_prototype = zeros(ndiff)
    )
    # The number of boundary equations matches the differential dimension,
    # while SciMLBase's automatic nlls inference compares against all variables.
    return BVProblem(fun, copy(u0), tspan, p; nlls = Val(false))
end

function test_dae_solution(sol, fixture; differential_tol, algebraic_tol, constraint_tol)
    (; exact, f, bca, bcb, p, tspan, ndiff) = fixture
    @test successful_retcode(sol)
    @test all(isfinite, Array(sol))
    ts = [collect(range(tspan...; length = 101)); first(tspan) + 0.371 * (last(tspan) - first(tspan))]
    differential_error = maximum(
        norm(sol(t)[1:ndiff] - exact(t)[1:ndiff], Inf) for t in ts
    )
    algebraic_error = maximum(
        norm(sol(t)[(ndiff + 1):end] - exact(t)[(ndiff + 1):end], Inf) for t in ts
    )
    constraint_error = maximum(norm(f(sol(t), p, t)[(ndiff + 1):end], Inf) for t in ts)
    @test differential_error < differential_tol
    @test algebraic_error < algebraic_tol
    @test constraint_error < constraint_tol
    @test maximum(norm(u[1:ndiff] - exact(t)[1:ndiff], Inf) for (u, t) in zip(sol.u, sol.t)) < differential_tol
    @test maximum(norm(u[(ndiff + 1):end] - exact(t)[(ndiff + 1):end], Inf) for (u, t) in zip(sol.u, sol.t)) < algebraic_tol
    @test norm([bca(sol(first(tspan)), p); bcb(sol(last(tspan)), p)], Inf) < 1.0e-8
    return (; differential_error, algebraic_error, constraint_error)
end

# Ascher & Spiteri, TR-92-18, Section 5.1, Example 1 with p(t)=0:
# https://www.cs.ubc.ca/sites/default/files/tr/1992/TR-92-18.pdf
# The original x1(0)=1 is replaced by x1(1)=e to make this a two-endpoint
# problem; the initial compatibility condition x1(0)-2x2(0)=-1 is retained.
function variable_dae_f(u, nu, t)
    x1, x2, y = u
    return [
        (nu - inv(2 - t)) * x1 + (2 - t) * nu * y + (3 - t) / (2 - t) * exp(t),
        (nu - 1) / (2 - t) * x1 - x2 + (nu - 1) * y + 2exp(t),
        (t + 2) * x1 + (t^2 - 4) * x2 - (t^2 + t - 2) * exp(t),
    ]
end
function variable_dae(nu)
    return (;
        f = variable_dae_f,
        bca = (u, p) -> [u[1] - 2u[2] + 1], bcb = (u, p) -> [u[1] - exp(1.0)],
        exact = t -> [exp(t), exp(t), -exp(t) / (2 - t)],
        u0 = zeros(3), tspan = (0.0, 1.0), p = nu, ndiff = 2,
    )
end

# Scalar specialization of the optimality system in Example 1.5 of
# "Local and Global Canonical Forms for Differential-Algebraic Equations
# with Symmetries", https://doi.org/10.1007/s10013-022-00596-x:
# min 1/2*integral(nu^2*x^2 + control^2), x'=control, x(0)=1.
# The costate has lambda(1)=0; stationarity gives control+lambda=0.
control_dae_f(u, nu, t) = [u[3], -nu^2 * u[1], u[2] + u[3]]
function control_dae_exact(t, nu)
    # Equivalent to cosh(nu*(1-t))/cosh(nu), avoiding growing exponentials.
    a, b = exp(-nu * t), exp(-nu * (2 - t))
    denom = 1 + exp(-2nu)
    costate = nu * (a - b) / denom
    return [(a + b) / denom, costate, -costate]
end
function control_dae(nu)
    return (;
        f = control_dae_f,
        bca = (u, p) -> [u[1] - 1], bcb = (u, p) -> [u[2]],
        exact = t -> control_dae_exact(t, nu),
        u0 = zeros(3), tspan = (0.0, 1.0), p = nu, ndiff = 2,
    )
end

# Bratu's BVP, also used in the official scipy.integrate.solve_bvp examples:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html
# Reformulate x''+lambda*exp(x)=0 as an index-1 DAE by retaining z=exp(x).
# lambda=2*a^2/cosh(a/2)^2 gives the exact branch below, with a=2.
bratu_dae_f(u, lambda, t) = [u[2], -lambda * u[3], u[3] - exp(u[1])]
function bratu_dae_exact(t)
    a = 2.0
    ratio = cosh(a * (t - 0.5)) / cosh(a / 2)
    return [-2log(ratio), -2a * tanh(a * (t - 0.5)), inv(ratio)^2]
end
bratu_dae() = (;
    f = bratu_dae_f,
    bca = (u, p) -> [u[1]], bcb = (u, p) -> [u[1]],
    exact = bratu_dae_exact, u0 = zeros(3), tspan = (0.0, 1.0),
    p = 8 / cosh(1)^2, ndiff = 2,
)

# The electrical-circuit DAE in Example 1.3 of the same canonical-forms paper.
# Set L=C1=C2=RG=RL=RR=1 and the voltage source to zero. Unknowns are
# (I,V1,V2,IG,IR), with two distinct algebraic current constraints.
function circuit_dae_f(u, p, t)
    current, v1, v2, ig, ir = u
    return [-current - v1 + v2, current - ig, -current - ir, v1 - ig, v2 - ir]
end
function circuit_dae_exact(t)
    current = exp(-t) * cos(sqrt(2) * t)
    voltage = exp(-t) * sin(sqrt(2) * t) / sqrt(2)
    return [current, voltage, -voltage, voltage, -voltage]
end
function circuit_dae()
    voltage = circuit_dae_exact(1.0)[2]
    return (;
        f = circuit_dae_f, bca = (u, p) -> [u[1] - 1],
        bcb = (u, p) -> [u[2] - voltage, u[3] + voltage],
        exact = circuit_dae_exact, u0 = zeros(5), tspan = (0.0, 1.0),
        p = nothing, ndiff = 3,
    )
end

# Original Ascher--Spiteri Section 5.5, Example 3: parameter estimation with
# six differential variables (x1,x2,w,lambda1,lambda2,v) and two index-2
# constraints. Smooth observations recover w=pi/3; the adjoint and algebraic
# variables vanish.
function estimation_dae_f(u, p, t)
    x1, x2, w, l1, l2, v, y, mu = u
    omega = pi / 3
    mismatch = x1 + x2 - (sin(omega * t) / omega + cos(omega * t))
    return [
        x2 + x1 * y, -w^2 * x1 + x2 * y, 0,
        -y * l1 + w^2 * l2 - 2omega^2 * x1 * mu - mismatch,
        -l1 - y * l2 - 2x2 * mu - mismatch, 2w * x1 * l2,
        omega^2 * x1^2 + x2^2 - 1, x1 * l1 + x2 * l2,
    ]
end
function estimation_dae_exact(t)
    omega = pi / 3
    return [sin(omega * t) / omega, cos(omega * t), omega, 0, 0, 0, 0, 0]
end
estimation_dae() = (;
    f = estimation_dae_f,
    bca = (u, p) -> [u[1], u[2] - 1, u[6], u[5]],
    bcb = (u, p) -> [u[6], u[2] * u[4] - (pi / 3)^2 * u[1] * u[5]],
    exact = estimation_dae_exact,
    u0 = [0.5, 0.5, 1.0, 0, 0, 0, 0, 0], tspan = (0.0, 2.0), p = nothing, ndiff = 6,
)

@testset "Additional boundary DAE benchmarks" begin
    @testset "Analytical reference solutions" begin
        for fixture in (
                variable_dae(1.0), variable_dae(10.0), control_dae(1.0),
                control_dae(20.0), bratu_dae(), circuit_dae(), estimation_dae(),
            )
            (; exact, f, bca, bcb, p, tspan, ndiff) = fixture
            for t in range(tspan...; length = 5)
                rhs = f(exact(t), p, t)
                derivative = ForwardDiff.derivative(exact, t)
                @test rhs[1:ndiff] ≈ derivative[1:ndiff] atol = 1.0e-12
                @test norm(rhs[(ndiff + 1):end], Inf) < 1.0e-12
            end
            @test norm([bca(exact(first(tspan)), p); bcb(exact(last(tspan)), p)], Inf) < 1.0e-12
        end
    end

    @testset "All Ascher stages on index-1 optimal control" begin
        fixture = control_dae(1.0)
        prob = dae_problem(fixture, false, false)
        differential_tols = (2.0e-3, 1.0e-5, 1.0e-7, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9)
        algebraic_tols = (0.1, 1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9, 1.0e-9, 1.0e-9)
        for (stage, alg) in enumerate((Ascher1(), Ascher2(), Ascher3(), Ascher4(), Ascher5(), Ascher6(), Ascher7()))
            sol = solve(prob, alg; dt = 0.1, adaptive = false, abstol = 1.0e-11)
            test_dae_solution(
                sol, fixture; differential_tol = differential_tols[stage],
                algebraic_tol = algebraic_tols[stage], constraint_tol = algebraic_tols[stage]
            )
        end
    end

    @testset "Index-1 BVP and TwoPointBVProblem interfaces" begin
        for fixture in (control_dae(1.0), bratu_dae(), circuit_dae()),
                inplace in (false, true), twopoint in (false, true)
            @testset "$(fixture.f), inplace=$inplace, twopoint=$twopoint" begin
                sol = solve(
                    dae_problem(fixture, inplace, twopoint), Ascher4();
                    dt = 0.1, adaptive = false, abstol = 1.0e-11
                )
                test_dae_solution(
                    sol, fixture; differential_tol = 1.0e-6,
                    algebraic_tol = 5.0e-5, constraint_tol = 6.0e-5
                )
            end
        end
    end

    @testset "Variable-coefficient index-2 DAE" begin
        for nu in (1.0, 5.0, 10.0), inplace in (false, true), twopoint in (false, true)
            fixture = variable_dae(nu)
            @testset "nu=$nu, inplace=$inplace, twopoint=$twopoint" begin
                # Unprojected even-stage collocation becomes less stable as
                # nu grows; the odd-stage method covers the harder parameter.
                alg = nu == 10.0 ? Ascher5() : Ascher4()
                sol = solve(
                    dae_problem(fixture, inplace, twopoint), alg;
                    dt = 0.05, adaptive = false, abstol = 1.0e-11
                )
                test_dae_solution(
                    sol, fixture; differential_tol = 2.0e-5,
                    algebraic_tol = 2.0e-5, constraint_tol = 1.0e-4
                )
            end
        end
    end

    @testset "Nonlinear index-2 parameter estimation" begin
        fixture = estimation_dae()
        for inplace in (false, true), twopoint in (false, true)
            @testset "inplace=$inplace, twopoint=$twopoint" begin
                sol = solve(
                    dae_problem(fixture, inplace, twopoint), Ascher4();
                    dt = 0.1, adaptive = false, abstol = 1.0e-10,
                    nlsolve_kwargs = (; abstol = 1.0e-10, maxiters = 100)
                )
                test_dae_solution(
                    sol, fixture; differential_tol = 1.0e-6,
                    algebraic_tol = 1.0e-6, constraint_tol = 1.0e-6
                )
                @test maximum(abs(u[3] - pi / 3) for u in sol.u) < 1.0e-7
            end
        end
    end

    @testset "Mesh convergence of differential and algebraic variables" begin
        for fixture in (bratu_dae(), variable_dae(5.0))
            prob = dae_problem(fixture, false, false)
            errors = map((0.1, 0.05)) do dt
                sol = solve(prob, Ascher4(); dt, adaptive = false, abstol = 1.0e-11)
                @test successful_retcode(sol)
                ts = range(fixture.tspan...; length = 201)
                return (
                    maximum(norm(sol(t)[1:fixture.ndiff] - fixture.exact(t)[1:fixture.ndiff], Inf) for t in ts),
                    maximum(norm(sol(t)[(fixture.ndiff + 1):end] - fixture.exact(t)[(fixture.ndiff + 1):end], Inf) for t in ts),
                )
            end
            # A factor of at least eight avoids pinning a platform-dependent
            # fitted order, but catches loss of convergence in either component.
            @test errors[2][1] < errors[1][1] / 8
            @test errors[2][2] < errors[1][2] / 8
        end
    end

    @testset "Adaptive optimal-control boundary layer" begin
        fixture = control_dae(20.0)
        for inplace in (false, true), twopoint in (false, true)
            prob = dae_problem(fixture, inplace, twopoint)
            sol = solve(prob, Ascher4(); dt = 0.1, adaptive = true, abstol = 1.0e-8)
            test_dae_solution(
                sol, fixture; differential_tol = 2.0e-8,
                algebraic_tol = 2.0e-6, constraint_tol = 2.0e-6
            )
            @test length(sol.t) > 11
        end
    end
end
