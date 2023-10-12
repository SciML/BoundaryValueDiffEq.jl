using Test, SafeTestsets

@testset "Boundary Value Problem Tests" begin
    @time @testset "Shooting Method Tests" begin
        @time @safetestset "Shooting Tests" begin
            include("shooting_tests.jl")
        end
        @time @safetestset "Orbital" begin
            include("orbital.jl")
        end
    end

    @time @testset "Collocation Method (MIRK) Tests" begin
        @time @safetestset "Ensemble" begin
            include("ensemble.jl")
        end
        @time @safetestset "MIRK Convergence Tests" begin
            include("mirk_convergence_tests.jl")
        end
        @time @safetestset "Vector of Vector" begin
            include("vectorofvector_initials.jl")
        end
    end

    @time @testset "ODE Interface Solvers" begin
        @time @safetestset "ODE Interface Tests" begin
            include("odeinterface_ex7.jl")
        end
    end

    @time @testset "Non Vector Inputs Tests" begin
        @time @safetestset "Non Vector Inputs" begin
            include("non_vector_inputs.jl")
        end
    end
    
    @time @testset "Interpolation Tests" begin
        @time @safetestset "MIRK Interpolation Test" begin
            include("interpolation_test.jl")
        end
    end
end
