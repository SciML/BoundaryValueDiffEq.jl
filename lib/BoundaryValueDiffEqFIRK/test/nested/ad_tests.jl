@testitem "Different AD compatibility" begin
    using BoundaryValueDiffEqFIRK
    using ForwardDiff, Enzyme, Mooncake

    @testset "Test different AD on multipoint BVP" begin
        function simplependulum!(du, u, p, t)
            θ = u[1]
            dθ = u[2]
            du[1] = dθ
            du[2] = -9.81 * sin(θ)
        end
        function bc!(residual, u, p, t)
            residual[1] = u[:, end ÷ 2][1] + pi / 2
            residual[2] = u[:, end][1] - pi / 2
        end
        u0 = [pi / 2, pi / 2]
        tspan = (0.0, pi / 2)
        prob = BVProblem(simplependulum!, bc!, u0, tspan)
        jac_alg_forwarddiff = BVPJacobianAlgorithm(
            bc_diffmode = AutoSparse(AutoForwardDiff()), nonbc_diffmode = AutoForwardDiff())
        jac_alg_enzyme = BVPJacobianAlgorithm(
            bc_diffmode = AutoSparse(AutoEnzyme(
                mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated)),
            nonbc_diffmode = AutoEnzyme(
                mode = Enzyme.Forward, function_annotation = Enzyme.Duplicated))
        jac_alg_mooncake = BVPJacobianAlgorithm(
            bc_diffmode = AutoSparse(AutoMooncake(; config = nothing)),
            nonbc_diffmode = AutoEnzyme(
                mode = Enzyme.Forward, function_annotation = Enzyme.Duplicated))
        for jac_alg in [jac_alg_forwarddiff, jac_alg_enzyme, jac_alg_mooncake]
            @test_nowarn sol = solve(
                prob, RadauIIa5(; jac_alg = jac_alg, nested_nlsolve = true), dt = 0.05)
        end
    end
    #=
        @testset "Test different AD on multipoint BVP using Interpolation BC" begin
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
            tspan = (0.0, pi / 2)
            prob = BVProblem(simplependulum!, bc!, u0, tspan)
            jac_alg_forwarddiff = BVPJacobianAlgorithm(
                bc_diffmode = AutoSparse(AutoForwardDiff()), nonbc_diffmode = AutoForwardDiff())
            jac_alg_enzyme = BVPJacobianAlgorithm(
                bc_diffmode = AutoSparse(AutoEnzyme(
                    mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated)),
                nonbc_diffmode = AutoEnzyme(
                    mode = Enzyme.Forward, function_annotation = Enzyme.Duplicated))
            jac_alg_mooncake = BVPJacobianAlgorithm(
                bc_diffmode = AutoSparse(AutoMooncake(; config = nothing)),
                nonbc_diffmode = AutoEnzyme(
                    mode = Enzyme.Forward, function_annotation = Enzyme.Duplicated))
            for jac_alg in [jac_alg_forwarddiff, jac_alg_enzyme, jac_alg_mooncake]
                @test_nowarn sol = solve(prob, RadauIIa5(; jac_alg = jac_alg, nested_nlsolve = true), dt = 0.05)
            end
        end
    =#
    @testset "Test different AD on twopoint BVP" begin
        function f!(du, u, p, t)
            du[1] = u[2]
            du[2] = 0
        end
        function boundary_two_point_a!(resida, ua, p)
            resida[1] = ua[1] - 5
        end
        function boundary_two_point_b!(residb, ub, p)
            residb[1] = ub[1]
        end

        odef! = ODEFunction(f!, analytic = (u0, p, t) -> [5 - t, -1])
        bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))
        tspan = (0.0, 5.0)
        u0 = [5.0, -3.5]
        prob = TwoPointBVProblem(odef!, (boundary_two_point_a!, boundary_two_point_b!),
            u0, tspan; bcresid_prototype, nlls = Val(false))
        jac_alg_forwarddiff = BVPJacobianAlgorithm(AutoSparse(AutoForwardDiff()))
        jac_alg_enzyme = BVPJacobianAlgorithm(AutoSparse(AutoEnzyme(
            mode = Enzyme.Forward, function_annotation = Enzyme.Duplicated)))
        jac_alg_mooncake = BVPJacobianAlgorithm(AutoSparse(AutoMooncake(;
            config = nothing)))
        for jac_alg in [jac_alg_forwarddiff, jac_alg_enzyme, jac_alg_mooncake]
            @test_nowarn sol = solve(
                prob, RadauIIa5(; jac_alg = jac_alg, nested_nlsolve = true), dt = 0.01)
        end
    end
end
