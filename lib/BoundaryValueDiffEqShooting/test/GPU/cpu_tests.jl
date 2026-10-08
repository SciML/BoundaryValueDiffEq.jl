include("device_tests.jl")

@testset "CPU ODE algorithms without DiffEqGPU" begin
    @test Base.get_extension(DeviceShooting, :BoundaryValueDiffEqShootingDiffEqGPUExt) === nothing
    # A user-supplied Tsit5 limiter must reach the OrdinaryDiffEq integrator.
    calls = Threads.Atomic{Int}(0)
    limiter! = (u, integrator, p, t) -> (Threads.atomic_add!(calls, 1); nothing)
    prob = TwoPointBVProblem(
        device_oscillator!, (device_left!, device_right!), [0.1, 0.9], (0.0, 1.0);
        bcresid_prototype = (zeros(1), zeros(1))
    )
    alg = MultipleShooting(4, Tsit5(; stage_limiter! = limiter!); device_steps = 8)
    sol = solve(prob, alg; abstol = 1.0e-9)
    @test successful_retcode(sol)
    @test calls[] > 0
    @test sol.u[end] ≈ [sin(1), cos(1)] atol = 1.0e-8
end

device_shooting_tests(CPU())
