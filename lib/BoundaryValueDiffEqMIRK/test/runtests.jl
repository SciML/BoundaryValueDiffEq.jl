using SafeTestsets, Test
using SciMLTesting

const MIRK_AD_ENV = joinpath(@__DIR__, "AD")

mirk_ad_all() = @time @safetestset "MIRK AD Tests" include("AD/ad_tests.jl")
mirk_ad_multipoint_grid() = @time @safetestset "MIRK AD Multipoint Grid Tests" begin
    withenv("BOUNDARYVALUEDIFFEQ_MIRK_AD_GROUP" => "MULTIPOINT_GRID") do
        include("AD/ad_tests.jl")
    end
end
mirk_ad_multipoint_interpolation() = @time @safetestset "MIRK AD Multipoint Interpolation Tests" begin
    withenv("BOUNDARYVALUEDIFFEQ_MIRK_AD_GROUP" => "MULTIPOINT_INTERPOLATION") do
        include("AD/ad_tests.jl")
    end
end
mirk_ad_twopoint() = @time @safetestset "MIRK AD TwoPoint Tests" begin
    withenv("BOUNDARYVALUEDIFFEQ_MIRK_AD_GROUP" => "TWOPOINT") do
        include("AD/ad_tests.jl")
    end
end

run_tests(;
    env = "BOUNDARYVALUEDIFFEQ_TEST_GROUP",
    core = function ()
        @time @safetestset "MIRK Basic Tests" include("Core/mirk_basic_tests.jl")
        @time @safetestset "MIRK NLLS Tests" include("Core/nlls_tests.jl")
        @time @safetestset "MIRK Ensemble Tests" include("Core/ensemble_tests.jl")
        @time @safetestset "MIRK Singular BVP Tests" include("Core/singular_bvp_tests.jl")
        @time @safetestset "MIRK DAE Tests" include("Core/dae_tests.jl")
        @time @safetestset "MIRK VectorOfVector Initials Tests" include("Core/vectorofvector_initials_tests.jl")
        return @time @safetestset "MIRK Dynamic Optimization Tests" include("Core/dynamic_optimization_tests.jl")
    end,
    groups = Dict(
        # AD: the different-AD-backend compatibility tests. Enzyme and Mooncake are
        # heavy optional backends kept out of the main test environment (they force a
        # large joint at-floor resolve on the Downgrade lane); they live in this
        # group's own test/AD/Project.toml, auto-activated before the body runs.
        "AD" => (;
            env = MIRK_AD_ENV,
            body = mirk_ad_all,
        ),
        "AD_MULTIPOINT_GRID" => (;
            env = MIRK_AD_ENV,
            body = mirk_ad_multipoint_grid,
        ),
        "AD_MULTIPOINT_INTERPOLATION" => (;
            env = MIRK_AD_ENV,
            body = mirk_ad_multipoint_interpolation,
        ),
        "AD_TWOPOINT" => (;
            env = MIRK_AD_ENV,
            body = mirk_ad_twopoint,
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
    all = ["Core", "AD_MULTIPOINT_GRID", "AD_MULTIPOINT_INTERPOLATION", "AD_TWOPOINT", "QA"],
)
