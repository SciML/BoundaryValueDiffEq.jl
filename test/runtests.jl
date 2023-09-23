using Test, SafeTestsets

@testset "Boundary Value Problem Tests" begin
    @time @testset "Shooting Method Tests" begin
        @time @safetestset "Shooting Tests" begin
            include("shooting/shooting_tests.jl")
        end
        @time @safetestset "Orbital" begin
            include("shooting/orbital.jl")
        end
    end

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
    end

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
