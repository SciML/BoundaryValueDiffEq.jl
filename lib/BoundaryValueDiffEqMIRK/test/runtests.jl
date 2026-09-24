using SafeTestsets, Test
using SciMLTesting

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        @time @safetestset "MIRK Basic Tests" include("Core/mirk_basic_tests.jl")
        @time @safetestset "MIRK Almost Banded Tests" include("Core/almost_banded_tests.jl")
        @time @safetestset "MIRK NLLS Tests" include("Core/nlls_tests.jl")
        @time @safetestset "MIRK Ensemble Tests" include("Core/ensemble_tests.jl")
        @time @safetestset "MIRK Singular BVP Tests" include("Core/singular_bvp_tests.jl")
        @time @safetestset "MIRK VectorOfVector Initials Tests" include("Core/vectorofvector_initials_tests.jl")
        return @time @safetestset "MIRK Dynamic Optimization Tests" include("Core/dynamic_optimization_tests.jl")
    end,
    groups = Dict(
        # Exercise the portable device kernels in ordinary CI without extending
        # the existing Core lane, which already runs near its time budget.
        "DeviceKernels" => function ()
            @time @safetestset "MIRK Packed Collocation Tests" include("GPU/packed_collocation_tests.jl")
            @time @safetestset "MIRK Resident Device Tests" include("GPU/resident_tests.jl")
            return @time @safetestset "MIRK Device Interpolation Tests" include("GPU/device_interpolation_tests.jl")
        end,
        # Keep the existing DAE group separate from the long-running Core suite.
        "DAE" => function ()
            return @time @safetestset "MIRK DAE Tests" include("Core/dae_tests.jl")
        end,
        # CUDA is optional and stays out of the normal test/dependency environment.
        # Run explicitly with BOUNDARYVALUEDIFFEQ_TEST_GROUP=GPU on a CUDA host.
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                return @time @safetestset "MIRK CUDA Tests" include("GPU/cuda_tests.jl")
            end,
        ),
        # AD: the different-AD-backend compatibility tests. Enzyme and Mooncake are
        # heavy optional backends kept out of the main test environment (they force a
        # large joint at-floor resolve on the Downgrade lane); they live in this
        # group's own test/AD/Project.toml, auto-activated before the body runs.
        "AD" => (;
            env = joinpath(@__DIR__, "AD"),
            body = function ()
                return @time @safetestset "MIRK AD Tests" include("AD/ad_tests.jl")
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
    all = ["Core", "DAE", "DeviceKernels", "AD", "QA"],
)
