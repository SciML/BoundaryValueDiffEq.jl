@testitem "Verbose field in caches" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: @SciMLMessage, SciMLLogging

    # Simple pendulum problem for testing
    tspan = (0.0, pi / 2)
    function simplependulum!(du, u, p, t)
        θ = u[1]
        dθ = u[2]
        du[1] = dθ
        du[2] = -9.81 * sin(θ)
    end
    function bc!(residual, u, p, t)
        residual[1] = u(pi / 4)[1] + pi / 2
        residual[2] = u(pi / 2)[1] - pi / 2
    end
    u0 = [pi / 2, pi / 2]
    prob = BVProblem(simplependulum!, bc!, u0, tspan)

    # Test MIRK with default verbose
    @testset "MIRK default verbose" begin
        cache = init(prob, MIRK4(), dt = 0.05)
        @test cache.verbose isa BVPVerbosity
        @test cache.verbose == BVPVerbosity()  # Default should be Standard preset
    end

    # Test MIRK with verbose = false
    @testset "MIRK verbose = false" begin
        cache = init(prob, MIRK4(), dt = 0.05, verbose = false)
        @test cache.verbose isa BVPVerbosity
        @test cache.verbose == BVPVerbosity(SciMLLogging.None())
    end

    # Test MIRK with verbose = true
    @testset "MIRK verbose = true" begin
        cache = init(prob, MIRK4(), dt = 0.05, verbose = true)
        @test cache.verbose isa BVPVerbosity
        @test cache.verbose == BVPVerbosity()  # true should give Standard preset
    end

    # Test MIRK with BVPVerbosity preset
    @testset "MIRK verbose = BVPVerbosity(Detailed())" begin
        cache = init(prob, MIRK4(), dt = 0.05, verbose = BVPVerbosity(SciMLLogging.Detailed()))
        @test cache.verbose isa BVPVerbosity
        @test cache.verbose == BVPVerbosity(SciMLLogging.Detailed())
    end

    # Test FIRK (nested) with verbose
    @testset "FIRK nested verbose" begin
        cache = init(prob, LobattoIIIa4(nested_nlsolve = true), dt = 0.05)
        @test cache.verbose isa BVPVerbosity

        cache_false = init(prob, LobattoIIIa4(nested_nlsolve = true), dt = 0.05, verbose = false)
        @test cache_false.verbose == BVPVerbosity(SciMLLogging.None())
    end

    # Test FIRK (expanded) with verbose
    @testset "FIRK expanded verbose" begin
        cache = init(prob, LobattoIIIa4(nested_nlsolve = false), dt = 0.05)
        @test cache.verbose isa BVPVerbosity

        cache_false = init(prob, LobattoIIIa4(nested_nlsolve = false), dt = 0.05, verbose = false)
        @test cache_false.verbose == BVPVerbosity(SciMLLogging.None())
    end

    # Test __split_kwargs with verbose
    @testset "__split_kwargs verbose extraction" begin
        using BoundaryValueDiffEqCore: __split_kwargs, DEFAULT_VERBOSE

        # Test with default verbose
        (abstol, adaptive, controller, verbose), remaining = __split_kwargs(
            abstol = 1e-6, adaptive = true, controller = DefectControl()
        )
        @test abstol == 1e-6
        @test adaptive == true
        @test controller isa DefectControl
        @test verbose == DEFAULT_VERBOSE

        # Test with custom verbose
        custom_verbose = BVPVerbosity(SciMLLogging.None())
        (abstol, adaptive, controller, verbose), remaining = __split_kwargs(
            abstol = 1e-4, adaptive = false, controller = NoErrorControl(),
            verbose = custom_verbose
        )
        @test abstol == 1e-4
        @test adaptive == false
        @test controller isa NoErrorControl
        @test verbose == custom_verbose
    end
end

@testitem "Verbose field in MIRKN cache" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: @SciMLMessage, SciMLLogging

    # Second order BVP for MIRKN
    tspan = (0.0, 1.0)
    function f!(ddu, du, u, p, t)
        ddu[1] = -u[1]
    end
    function bc!(res, du, u, p, t)
        res[1] = u[1][1]
        res[2] = u[end][1] - 1
    end
    u0 = [0.0]
    prob = SecondOrderBVProblem(f!, bc!, u0, tspan)

    # Test MIRKN with default verbose
    @testset "MIRKN default verbose" begin
        cache = init(prob, MIRKN4(), dt = 0.1)
        @test cache.verbose isa BVPVerbosity
        @test cache.verbose == BVPVerbosity()
    end

    # Test MIRKN with verbose = false
    @testset "MIRKN verbose = false" begin
        cache = init(prob, MIRKN4(), dt = 0.1, verbose = false)
        @test cache.verbose == BVPVerbosity(SciMLLogging.None())
    end

    # Test MIRKN with custom verbose
    @testset "MIRKN custom verbose" begin
        cache = init(prob, MIRKN4(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.All()))
        @test cache.verbose == BVPVerbosity(SciMLLogging.All())
    end
end

@testitem "Verbose persistence through solve" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: @SciMLMessage, SciMLLogging

    # Simple test problem
    tspan = (0.0, 1.0)
    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end
    function bc!(res, u, p, t)
        res[1] = u[1][1]
        res[2] = u[end][1] - 1
    end
    u0 = [0.0, 1.0]
    prob = BVProblem(f!, bc!, u0, tspan)

    # Test that verbose setting persists through the solve
    @testset "Verbose persists in MIRK solve" begin
        sol = solve(prob, MIRK4(), dt = 0.1, verbose = false)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Verbose persists in FIRK solve" begin
        sol = solve(prob, RadauIIa5(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.None()))
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Verbose with All preset" begin
        # This should work without errors even with maximum verbosity
        sol = solve(prob, MIRK4(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.All()))
        @test SciMLBase.successful_retcode(sol)
    end
end

@testitem "NonlinearVerbosity integration" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: SciMLLogging
    using NonlinearSolveFirstOrder: NonlinearVerbosity

    # Test default includes NonlinearVerbosity
    @testset "Default nonlinear_verbosity" begin
        v = BVPVerbosity()
        @test hasfield(typeof(v), :nonlinear_verbosity)
        @test v.nonlinear_verbosity == SciMLLogging.Standard()
    end

    # Test preset matching
    @testset "Preset matching" begin
        v_none = BVPVerbosity(SciMLLogging.None())
        @test v_none.nonlinear_verbosity == SciMLLogging.None()

        v_detailed = BVPVerbosity(SciMLLogging.Detailed())
        @test v_detailed.nonlinear_verbosity == SciMLLogging.Detailed()

        v_all = BVPVerbosity(SciMLLogging.All())
        @test v_all.nonlinear_verbosity == SciMLLogging.All()
    end

    # Test that Standard preset sets appropriate nonlinear_verbosity
    @testset "Standard preset nonlinear_verbosity" begin
        v = BVPVerbosity(SciMLLogging.Standard())
        @test v.bvpsol_convergence == SciMLLogging.WarnLevel()  # Standard preset
        @test v.nonlinear_verbosity == SciMLLogging.Standard()  # Matches preset
    end

    # Test integration with solve - verify it doesn't error
    @testset "NonlinearVerbosity passed to solvers" begin
        # Simple test problem
        tspan = (0.0, 1.0)
        function f!(du, u, p, t)
            du[1] = u[2]
            du[2] = -u[1]
        end
        function bc!(res, u, p, t)
            res[1] = u[1][1]
            res[2] = u[end][1] - 1
        end
        u0 = [0.0, 1.0]
        prob = BVProblem(f!, bc!, u0, tspan)

        # Test MIRK with All preset (includes All nonlinear verbosity)
        sol = solve(prob, MIRK4(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.All())
        )
        @test SciMLBase.successful_retcode(sol)

        # Test FIRK with None preset (silences nonlinear verbosity)
        sol = solve(prob, RadauIIa5(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.None())
        )
        @test SciMLBase.successful_retcode(sol)
    end

    # Test with different solver types
    @testset "NonlinearVerbosity with various solvers" begin
        tspan = (0.0, 1.0)
        function f!(du, u, p, t)
            du[1] = u[2]
            du[2] = -u[1]
        end
        function bc!(res, u, p, t)
            res[1] = u[1][1]
            res[2] = u[end][1] - 1
        end
        u0 = [0.0, 1.0]
        prob = BVProblem(f!, bc!, u0, tspan)

        # MIRKN for second order BVP
        tspan2 = (0.0, 1.0)
        function f2!(ddu, du, u, p, t)
            ddu[1] = -u[1]
        end
        function bc2!(res, du, u, p, t)
            res[1] = u[1][1]
            res[2] = u[end][1] - 1
        end
        u0_2 = [0.0]
        prob2 = SecondOrderBVProblem(f2!, bc2!, u0_2, tspan2)

        sol2 = solve(prob2, MIRKN4(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.Detailed())
        )
        @test SciMLBase.successful_retcode(sol2)
    end
end

@testitem "OptimizationVerbosity integration" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: SciMLLogging


    # Test default includes OptimizationVerbosity
    @testset "Default optimization_verbosity" begin
        v = BVPVerbosity()
        @test hasfield(typeof(v), :optimization_verbosity)
        @test v.optimization_verbosity == SciMLLogging.Standard()
    end

    # Test preset matching
    @testset "Preset matching" begin
        v_none = BVPVerbosity(SciMLLogging.None())
        @test v_none.optimization_verbosity == SciMLLogging.None()

        v_detailed = BVPVerbosity(SciMLLogging.Detailed())
        @test v_detailed.optimization_verbosity == SciMLLogging.Detailed()

        v_all = BVPVerbosity(SciMLLogging.All())
        @test v_all.optimization_verbosity == SciMLLogging.All()
    end

    # Test that Standard preset sets appropriate optimization_verbosity
    @testset "Standard preset optimization_verbosity" begin
        v = BVPVerbosity(SciMLLogging.Standard())
        @test v.bvpsol_convergence == SciMLLogging.WarnLevel()  # Standard preset
        @test v.optimization_verbosity == SciMLLogging.Standard()  # Matches preset
    end

    # Test that presets correctly set both verbosity controls
    @testset "Both verbosity controls via preset" begin
        v_none = BVPVerbosity(SciMLLogging.None())
        @test v_none.nonlinear_verbosity == SciMLLogging.None()
        @test v_none.optimization_verbosity == SciMLLogging.None()

        v_all = BVPVerbosity(SciMLLogging.All())
        @test v_all.nonlinear_verbosity == SciMLLogging.All()
        @test v_all.optimization_verbosity == SciMLLogging.All()
    end
end
