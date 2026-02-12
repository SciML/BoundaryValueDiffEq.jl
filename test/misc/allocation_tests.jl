@testitem "Allocation Tests" tags = [:misc] begin
    using BoundaryValueDiffEqCore: interval, recursive_flatten!, recursive_unflatten!,
        __maybe_matmul!, diff!
    using LinearAlgebra

    @testset "interval" begin
        mesh = collect(0.0:0.1:1.0)
        t = 0.55

        interval(mesh, t) # warmup

        allocs = @allocated interval(mesh, t)
        @test allocs == 0
    end

    @testset "recursive_flatten!" begin
        y = [rand(2) for _ in 1:10]
        x = zeros(20)

        recursive_flatten!(x, y) # warmup

        allocs = @allocated recursive_flatten!(x, y)
        @test allocs == 0
    end

    @testset "recursive_unflatten!" begin
        y = [zeros(2) for _ in 1:10]
        x = rand(20)

        recursive_unflatten!(y, x) # warmup

        allocs = @allocated recursive_unflatten!(y, x)
        @test allocs == 0
    end

    @testset "__maybe_matmul!" begin
        A = Matrix(rand(4, 4))
        b = Vector(rand(4))
        c = Vector(zeros(4))

        __maybe_matmul!(c, A, b) # warmup

        allocs = @allocated __maybe_matmul!(c, A, b)
        @test allocs == 0
    end

    @testset "diff!" begin
        x = collect(0.0:0.1:1.0)
        dx = zeros(length(x) - 1)

        diff!(dx, x) # warmup

        allocs = @allocated diff!(dx, x)
        @test allocs == 0
    end
end
