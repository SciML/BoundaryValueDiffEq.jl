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
            abstol = 1.0e-6, adaptive = true, controller = DefectControl()
        )
        @test abstol == 1.0e-6
        @test adaptive == true
        @test controller isa DefectControl
        @test verbose == DEFAULT_VERBOSE

        # Test with custom verbose
        custom_verbose = BVPVerbosity(SciMLLogging.None())
        (abstol, adaptive, controller, verbose), remaining = __split_kwargs(
            abstol = 1.0e-4, adaptive = false, controller = NoErrorControl(),
            verbose = custom_verbose
        )
        @test abstol == 1.0e-4
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

@testitem "Internal solver verbosity integration" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: SciMLLogging

    # Test BVPVerbosity has both verbosity fields
    @testset "Verbosity fields exist" begin
        v = BVPVerbosity()
        @test hasfield(typeof(v), :nonlinear_verbosity)
        @test hasfield(typeof(v), :optimization_verbosity)
        @test v.nonlinear_verbosity == SciMLLogging.Standard()
        @test v.optimization_verbosity == SciMLLogging.Standard()
    end

    # Test preset matching for both verbosity controls
    @testset "Preset matching" begin
        v_none = BVPVerbosity(SciMLLogging.None())
        @test v_none.nonlinear_verbosity == SciMLLogging.None()
        @test v_none.optimization_verbosity == SciMLLogging.None()

        v_detailed = BVPVerbosity(SciMLLogging.Detailed())
        @test v_detailed.nonlinear_verbosity == SciMLLogging.Detailed()
        @test v_detailed.optimization_verbosity == SciMLLogging.Detailed()

        v_all = BVPVerbosity(SciMLLogging.All())
        @test v_all.nonlinear_verbosity == SciMLLogging.All()
        @test v_all.optimization_verbosity == SciMLLogging.All()
    end

    # Test that Standard preset sets appropriate verbosity
    @testset "Standard preset verbosity" begin
        v = BVPVerbosity(SciMLLogging.Standard())
        @test v.bvpsol_convergence == SciMLLogging.WarnLevel()
        @test v.nonlinear_verbosity == SciMLLogging.Standard()
        @test v.optimization_verbosity == SciMLLogging.Standard()
    end

    # Test integration with solve - verify it doesn't error
    @testset "Solve with different verbosity presets" begin
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

        # Test with All preset
        sol = solve(
            prob, MIRK4(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.All())
        )
        @test SciMLBase.successful_retcode(sol)

        # Test with None preset
        sol = solve(
            prob, RadauIIa5(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.None())
        )
        @test SciMLBase.successful_retcode(sol)

        # Test with MIRKN
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

        sol2 = solve(
            prob2, MIRKN4(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.Detailed())
        )
        @test SciMLBase.successful_retcode(sol2)
    end
end

@testitem "Verbosity propagation via cache inspection" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: __concrete_kwargs, SciMLLogging
    using NonlinearSolveFirstOrder: NewtonRaphson
    using NonlinearSolveBase: NonlinearVerbosity
    using OptimizationBase: OptimizationVerbosity

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

    # Test MIRK with NonlinearSolve verbosity
    @testset "MIRK cache NonlinearSolve verbosity" begin
        # Init with All verbosity
        cache = init(prob, MIRK4(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.All()))

        # Extract what would be passed to internal solver
        kwargs = __concrete_kwargs(
            cache.alg.nlsolve, cache.alg.optimize,
            cache.nlsolve_kwargs, cache.optimize_kwargs,
            cache.verbose
        )

        @test haskey(kwargs, :verbose)
        @test kwargs.verbose isa NonlinearVerbosity

        # Init with None verbosity
        cache_none = init(prob, MIRK4(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.None()))
        kwargs_none = __concrete_kwargs(
            cache_none.alg.nlsolve, cache_none.alg.optimize,
            cache_none.nlsolve_kwargs, cache_none.optimize_kwargs,
            cache_none.verbose
        )

        @test haskey(kwargs_none, :verbose)
        @test kwargs_none.verbose isa NonlinearVerbosity
    end

    # Test FIRK with NonlinearSolve verbosity
    @testset "FIRK cache NonlinearSolve verbosity" begin
        cache = init(prob, RadauIIa5(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.Detailed()))

        kwargs = __concrete_kwargs(
            cache.alg.nlsolve, cache.alg.optimize,
            cache.nlsolve_kwargs, cache.optimize_kwargs,
            cache.verbose
        )

        @test haskey(kwargs, :verbose)
        @test kwargs.verbose isa NonlinearVerbosity
    end

    # Test MIRKN with NonlinearSolve verbosity
    @testset "MIRKN cache NonlinearSolve verbosity" begin
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

        cache = init(prob2, MIRKN4(), dt = 0.1, verbose = BVPVerbosity(SciMLLogging.Standard()))

        kwargs = __concrete_kwargs(
            cache.alg.nlsolve, cache.alg.optimize,
            cache.nlsolve_kwargs, cache.optimize_kwargs,
            cache.verbose
        )

        @test haskey(kwargs, :verbose)
        @test kwargs.verbose isa NonlinearVerbosity
    end

    # Test with explicit nlsolve algorithm
    @testset "Explicit nlsolve algorithm verbosity" begin
        cache = init(
            prob, MIRK4(nlsolve = NewtonRaphson()), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.All())
        )

        kwargs = __concrete_kwargs(
            cache.alg.nlsolve, cache.alg.optimize,
            cache.nlsolve_kwargs, cache.optimize_kwargs,
            cache.verbose
        )

        @test haskey(kwargs, :verbose)
        @test kwargs.verbose isa NonlinearVerbosity
    end

    # Test user-specified verbose takes precedence
    @testset "User nlsolve_kwargs verbose precedence" begin
        user_verbose = NonlinearVerbosity(SciMLLogging.Minimal())
        cache = init(
            prob, MIRK4(), dt = 0.1,
            verbose = BVPVerbosity(SciMLLogging.All()),
            nlsolve_kwargs = (; verbose = user_verbose)
        )

        kwargs = __concrete_kwargs(
            cache.alg.nlsolve, cache.alg.optimize,
            cache.nlsolve_kwargs, cache.optimize_kwargs,
            cache.verbose
        )

        @test kwargs.verbose === user_verbose
    end

    # Test OptimizationVerbosity extraction (mock optimization algorithm)
    @testset "OptimizationVerbosity extraction" begin
        # We can't easily test with a real optimization algorithm without adding dependencies
        # But we can test the __concrete_kwargs function directly with a mock optimize value
        bvp_verbose = BVPVerbosity(SciMLLogging.All())
        mock_opt_alg = :some_optimizer

        kwargs = __concrete_kwargs(
            nothing, mock_opt_alg,
            (;), (;),
            bvp_verbose
        )

        @test haskey(kwargs, :verbose)
        @test kwargs.verbose isa OptimizationVerbosity

        # Test with None preset
        bvp_verbose_none = BVPVerbosity(SciMLLogging.None())
        kwargs_none = __concrete_kwargs(
            nothing, mock_opt_alg,
            (;), (;),
            bvp_verbose_none
        )

        @test haskey(kwargs_none, :verbose)
        @test kwargs_none.verbose isa OptimizationVerbosity
    end

    # Test user-specified optimize_kwargs verbose takes precedence
    @testset "User optimize_kwargs verbose precedence" begin
        user_verbose = OptimizationVerbosity(SciMLLogging.Minimal())
        bvp_verbose = BVPVerbosity(SciMLLogging.All())
        mock_opt_alg = :some_optimizer

        kwargs = __concrete_kwargs(
            nothing, mock_opt_alg,
            (;), (; verbose = user_verbose),
            bvp_verbose
        )

        @test kwargs.verbose === user_verbose
    end
end

@testitem "Solve problems with various verbosity settings" begin
    using BoundaryValueDiffEq, BoundaryValueDiffEqCore
    using BoundaryValueDiffEqCore: SciMLLogging
    using NonlinearSolveFirstOrder: NewtonRaphson

    # Simple BVP problem
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

    # Test all presets with MIRK
    @testset "MIRK with all verbosity presets" begin
        for preset in [
                SciMLLogging.None(), SciMLLogging.Minimal(),
                SciMLLogging.Standard(), SciMLLogging.Detailed(),
                SciMLLogging.All(),
            ]
            sol = solve(prob, MIRK4(), dt = 0.1, verbose = BVPVerbosity(preset))
            @test SciMLBase.successful_retcode(sol)
        end
    end

    # Test with explicit nonlinear solver
    @testset "MIRK with explicit nlsolve and various verbosity" begin
        for preset in [SciMLLogging.None(), SciMLLogging.All()]
            sol = solve(
                prob, MIRK4(nlsolve = NewtonRaphson()), dt = 0.1,
                verbose = BVPVerbosity(preset)
            )
            @test SciMLBase.successful_retcode(sol)
        end
    end

    # Test FIRK with various verbosity
    @testset "FIRK with various verbosity" begin
        for preset in [SciMLLogging.None(), SciMLLogging.Standard(), SciMLLogging.All()]
            sol = solve(prob, RadauIIa5(), dt = 0.1, verbose = BVPVerbosity(preset))
            @test SciMLBase.successful_retcode(sol)

            sol = solve(prob, LobattoIIIa4(), dt = 0.1, verbose = BVPVerbosity(preset))
            @test SciMLBase.successful_retcode(sol)
        end
    end

    # Test MIRKN with various verbosity
    @testset "MIRKN with various verbosity" begin
        function f2!(ddu, du, u, p, t)
            ddu[1] = -u[1]
        end
        function bc2!(res, du, u, p, t)
            res[1] = u[1][1]
            res[2] = u[end][1] - 1
        end
        u0_2 = [0.0]
        prob2 = SecondOrderBVProblem(f2!, bc2!, u0_2, tspan)

        for preset in [
                SciMLLogging.None(), SciMLLogging.Minimal(),
                SciMLLogging.Standard(), SciMLLogging.All(),
            ]
            sol = solve(prob2, MIRKN4(), dt = 0.1, verbose = BVPVerbosity(preset))
            @test SciMLBase.successful_retcode(sol)
        end
    end

    # Test with Boolean verbose (backward compatibility)
    @testset "Boolean verbose backward compatibility" begin
        sol_true = solve(prob, MIRK4(), dt = 0.1, verbose = true)
        @test SciMLBase.successful_retcode(sol_true)

        sol_false = solve(prob, MIRK4(), dt = 0.1, verbose = false)
        @test SciMLBase.successful_retcode(sol_false)
    end
end
