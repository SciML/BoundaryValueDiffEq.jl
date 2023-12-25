using Test, SafeTestsets

const GROUP = uppercase(get(ENV, "GROUP", "ALL"))

@testset "Boundary Value Problem Tests" begin
    if GROUP == "ALL" || GROUP == "SHOOTING"
        @time @testset "Shooting Method Tests" begin
            @time @safetestset "Basic Problems" begin
                include("shooting/basic_problems.jl")
            end
            @time @safetestset "Orbital" begin
                include("shooting/orbital.jl")
            end
            @time @safetestset "Shooting NLLS Tests" begin
                include("shooting/nonlinear_least_squares.jl")
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
            if VERSION ≥ v"1.10-"
                @time @safetestset "MIRK NLLS Tests" begin
                    include("mirk/nonlinear_least_squares.jl")
                end
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
                include("misc/odeinterface_wrapper.jl")
            end
            @time @safetestset "Initial Guess Function" begin
                include("misc/initial_guess.jl")
            end
            @time @safetestset "Affine Geodesic" begin
                include("misc/affine_geodesic.jl")
            end
            @time @safetestset "Aqua: Quality Assurance" begin
                include("misc/aqua.jl")
            end
        end
    end
end
