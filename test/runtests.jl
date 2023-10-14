using Test, SafeTestsets

const GROUP = uppercase(get(ENV, "GROUP", "ALL"))

@testset "Boundary Value Problem Tests" begin
    if GROUP == "ALL" || GROUP == "SHOOTING"
        @time @testset "Shooting Method Tests" begin
            @time @safetestset "Shooting Tests" begin
                include("shooting/shooting_tests.jl")
            end
            @time @safetestset "Ray Tracing BVP" begin
                include("shooting/ray_tracing.jl")
            end
            @static if VERSION ≥ v"1.10.0-beta2"
                # Orbital Tests take extremely long to compile on Julia 1.9
                @time @safetestset "Orbital" begin
                    include("shooting/orbital.jl")
                end
            end
        end
    end

    if GROUP == "ALL" || GROUP == "MIRK"
        @time @testset "Collocation Method (MIRK) Tests" begin
            @time @safetestset "Ensemble" begin
                include("mirk/ensemble.jl")
            end
            @time @safetestset "MIRK Convergence Tests" begin
                include("mirk/mirk_convergence_tests.jl")
            end
            @time @safetestset "Vector of Vector" begin
                include("mirk/vectorofvector_initials.jl")
            end
            @time @safetestset "Interpolation Tests" begin
                include("mirk/interpolation_test.jl")
            end
        end
    end

    if GROUP == "ALL" || GROUP == "OTHERS"
        @time @testset "Miscelleneous" begin
            @time @safetestset "Non Vector Inputs" begin
                include("misc/non_vector_inputs.jl")
            end

            @time @safetestset "Type Stability" begin
                include("misc/type_stability.jl")
            end

            @time @safetestset "ODE Interface Tests" begin
                include("misc/odeinterface_ex7.jl")
            end
        end
    end
end
