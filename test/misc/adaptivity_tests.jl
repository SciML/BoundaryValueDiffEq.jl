@testitem "Mesh redistribution in adaptivity" begin
    using BoundaryValueDiffEq

    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -t / p * u[2] - pi^2 * cos(pi * t) - pi * t / p * sin(pi * t)
    end
    function bc!(res, u, p, t)
        res[1] = u[:, 1][1] + 2
        res[2] = u[:, end][1]
    end
    tspan = (-1.0, 1.0)
    sol = [1.0, 0.0]
    prob = BVProblem(f!, bc!, sol, tspan, 0.001)

    for order in (2, 3, 4, 5, 6)
        s = Symbol("MIRK$(order)")
        @eval mirk_solver(::Val{$order}) = $(s)()
    end
    for stage in (2, 3, 4, 5)
        s = Symbol("LobattoIIIa$(stage)")
        @eval lobattoIIIa_solver(
            ::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    for stage in (3, 4, 5)
        s = Symbol("LobattoIIIb$(stage)")
        @eval lobattoIIIb_solver(
            ::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    for stage in (3, 4, 5)
        s = Symbol("LobattoIIIc$(stage)")
        @eval lobattoIIIc_solver(
            ::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end

    for stage in (2, 3, 5, 7)
        s = Symbol("RadauIIa$(stage)")
        @eval radau_solver(::Val{$stage}, args...; kwargs...) = $(s)(args...; kwargs...)
    end
    @testset "MIRK$order method" for order in (2, 3, 4, 5, 6)
        @test_nowarn sol = solve(prob, mirk_solver(Val(order)), dt = 0.01)
    end

    @testset "RadauIIa$stage method" for stage in (2, 3, 5, 7)
        @test_nowarn sol = solve(prob, radau_solver(Val(stage)), dt = 0.01)
    end

    @testset "LobattoIIIa$stage method" for stage in (2, 3, 4, 5)
        @test_nowarn sol = solve(prob, lobattoIIIa_solver(Val(stage)), dt = 0.01)
    end

    @testset "LobattoIIIb$stage method" for stage in (3, 4, 5)
        @test_nowarn sol = solve(prob, lobattoIIIb_solver(Val(stage)), dt = 0.01)
    end

    @testset "LobattoIIIc$stage method" for stage in (3, 4, 5)
        @test_nowarn sol = solve(prob, lobattoIIIc_solver(Val(stage)), dt = 0.01)
    end
end
