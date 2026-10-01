using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    # Core: representative light FIRK set covering both formulations' basic solves.
    # Targeted by the uniform "Core" downgrade value. The full EXPANDED/NESTED groups
    # already include these basic tests, and "All" runs those groups, so "Core" is
    # kept out of "All" (see the `all` list below) to avoid double execution.
    core = function ()
        @time @safetestset "FIRK Public Facade Tests" include("public_facade_tests.jl")
        @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
        return @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
    end,
    groups = Dict(
        "PUBLIC_FACADE" => function ()
            return @time @safetestset "FIRK Public Facade Tests" include("public_facade_tests.jl")
        end,
        # The expanded formulation is split across several matrix groups. Nearly all of
        # its wall time is Julia compilation: every (problem, solver) pair specializes
        # the whole FIRK -> NonlinearSolve -> AD stack afresh at ~20 s a piece, and the
        # solver sweeps run 16 solvers over 4 problems. Run as one group it exceeds two
        # hours (issue #548), so the two solver sweeps get groups of their own.
        "EXPANDED_BASIC" => function ()
            return @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
        end,
        "EXPANDED_AFFINENESS" => function ()
            return @time @safetestset "FIRK Expanded Affineness Tests" include("expanded/firk_affineness_tests.jl")
        end,
        "EXPANDED_CONVERGENCE" => function ()
            return @time @safetestset "FIRK Expanded Convergence Tests" include("expanded/firk_convergence_tests.jl")
        end,
        "EXPANDED_NLLS" => function ()
            return @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
        end,
        "EXPANDED_MISC" => function ()
            @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
            @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
            @time @safetestset "FIRK Expanded DAE Tests" include("expanded/dae_tests.jl")
            return @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")
        end,
        # Aggregate of the five EXPANDED_* groups for running the whole formulation
        # locally in one process. Kept out of the CI matrix and out of `all` so it does
        # not duplicate them.
        "EXPANDED" => function ()
            @time @safetestset "FIRK Expanded Basic Tests" include("expanded/firk_basic_tests.jl")
            @time @safetestset "FIRK Expanded Affineness Tests" include("expanded/firk_affineness_tests.jl")
            @time @safetestset "FIRK Expanded Convergence Tests" include("expanded/firk_convergence_tests.jl")
            @time @safetestset "FIRK Expanded NLLS Tests" include("expanded/nlls_tests.jl")
            @time @safetestset "FIRK Expanded Ensemble Tests" include("expanded/ensemble_tests.jl")
            @time @safetestset "FIRK Expanded Singular BVP Tests" include("expanded/singular_bvp_tests.jl")
            @time @safetestset "FIRK Expanded DAE Tests" include("expanded/dae_tests.jl")
            return @time @safetestset "FIRK Expanded VectorOfVector Initials Tests" include("expanded/vectorofvector_initials_tests.jl")
        end,
        # The nested formulation is split for the same reason as the expanded one, but its
        # cost is not only compilation: the nested solvers run an inner nonlinear solve per
        # mesh point, so "Simple Pendulum" alone (17 solvers at dt = 0.005) takes ~55 min of
        # genuine solve time and gets its own group.
        "NESTED_BASIC" => function ()
            return @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
        end,
        "NESTED_AFFINENESS" => function ()
            return @time @safetestset "FIRK Nested Affineness Tests" include("nested/firk_affineness_tests.jl")
        end,
        "NESTED_CONVERGENCE" => function ()
            return @time @safetestset "FIRK Nested Convergence Tests" include("nested/firk_convergence_tests.jl")
        end,
        "NESTED_PENDULUM" => function ()
            return @time @safetestset "FIRK Nested Simple Pendulum Tests" include("nested/firk_pendulum_tests.jl")
        end,
        "NESTED_NLLS" => function ()
            return @time @safetestset "FIRK Nested NLLS Tests" include("nested/nlls_tests.jl")
        end,
        # Keep the aggregates for local runs, but partition the 22 expensive cases in CI.
        "NESTED_NLLS_UNDERCONSTRAINED" => function ()
            return @time @safetestset "FIRK Nested Underconstrained NLLS Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp()
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_NEWTON" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Newton Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(1:4)
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_GAUSS_NEWTON" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Gauss-Newton Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(5:8)
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_GAUSS_NEWTON_RADAU" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Gauss-Newton Radau Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(5:5)
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_GAUSS_NEWTON_LOBATTO_IIIA" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Gauss-Newton Lobatto IIIa Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(6:6)
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_GAUSS_NEWTON_LOBATTO_IIIB" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Gauss-Newton Lobatto IIIb Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(7:7)
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_GAUSS_NEWTON_LOBATTO_IIIC" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Gauss-Newton Lobatto IIIc Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(8:8)
            end
        end,
        "NESTED_NLLS_UNDERCONSTRAINED_TRUST_REGION" => function ()
            return @time @safetestset "FIRK Nested Underconstrained Trust-Region Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp(9:12)
            end
        end,
        "NESTED_MISC" => function ()
            @time @safetestset "FIRK Nested Ensemble Tests" include("nested/ensemble_tests.jl")
            @time @safetestset "FIRK Nested DAE Tests" include("nested/dae_tests.jl")
            return @time @safetestset "FIRK Nested VectorOfVector Initials Tests" include("nested/vectorofvector_initials_tests.jl")
        end,
        # Aggregate of the seven NESTED_* groups, for running the whole formulation locally
        # in one process. Kept out of the CI matrix and out of `all`.
        "NESTED" => function ()
            @time @safetestset "FIRK Nested Basic Tests" include("nested/firk_basic_tests.jl")
            @time @safetestset "FIRK Nested Affineness Tests" include("nested/firk_affineness_tests.jl")
            @time @safetestset "FIRK Nested Convergence Tests" include("nested/firk_convergence_tests.jl")
            @time @safetestset "FIRK Nested Simple Pendulum Tests" include("nested/firk_pendulum_tests.jl")
            @time @safetestset "FIRK Nested NLLS Tests" include("nested/nlls_tests.jl")
            @time @safetestset "FIRK Nested Underconstrained NLLS Tests" begin
                include("nested/nlls_underconstrained_tests.jl")
                test_underconstrained_bvp()
            end
            @time @safetestset "FIRK Nested Ensemble Tests" include("nested/ensemble_tests.jl")
            @time @safetestset "FIRK Nested DAE Tests" include("nested/dae_tests.jl")
            return @time @safetestset "FIRK Nested VectorOfVector Initials Tests" include("nested/vectorofvector_initials_tests.jl")
        end,
        # AD: the different-AD-backend compatibility tests. Enzyme and Mooncake are
        # heavy optional backends kept out of the main test environment (they force a
        # large joint at-floor resolve on the Downgrade lane); they live in this
        # group's own test/AD/Project.toml, auto-activated before the body runs.
        "DeviceKernels" => function ()
            @time @safetestset "FIRK flat buffer resizing" begin
                include("GPU/resizing_tests.jl")
                test_firk_flat_buffers(identity, CPU())
            end
            @time @safetestset "FIRK device kernels" include("GPU/device_backend_tests.jl")
            @time @safetestset "FIRK device regressions" include("GPU/device_regression_tests.jl")
            @time @safetestset "FIRK resident nested solves" include("GPU/device_nested_tests.jl")
            return @time @safetestset "FIRK device adaptivity and structure" include("GPU/device_features_tests.jl")
        end,
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                return @time @safetestset "FIRK CUDA" include("GPU/cuda_tests.jl")
            end,
        ),
        "AD" => (;
            env = joinpath(@__DIR__, "AD"),
            body = function ()
                return @time @safetestset "FIRK Expanded AD Tests" include("AD/ad_tests.jl")
            end,
        ),
    ),
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = function ()
            # QA (Aqua) runs on release + LTS Julia only; skip on prerelease.
            isempty(VERSION.prerelease) || return nothing
            return @time @safetestset "Quality Assurance" include("qa/qa.jl")
        end,
    ),
    # "All" runs the split EXPANDED_* and NESTED_* groups + AD + QA. "Core" and the
    # aggregate "EXPANDED"/"NESTED" groups are intentionally excluded: all three only
    # re-run tests that the listed groups already cover.
    all = [
        "PUBLIC_FACADE", "EXPANDED_BASIC", "EXPANDED_AFFINENESS", "EXPANDED_CONVERGENCE",
        "EXPANDED_NLLS", "EXPANDED_MISC",
        "NESTED_BASIC", "NESTED_AFFINENESS", "NESTED_CONVERGENCE",
        "NESTED_PENDULUM", "NESTED_NLLS", "NESTED_NLLS_UNDERCONSTRAINED", "NESTED_MISC",
        "DeviceKernels", "AD", "QA",
    ],
)
