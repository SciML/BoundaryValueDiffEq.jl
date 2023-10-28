module BoundaryValueDiffEqOrdinaryDiffEqExt

# This extension doesn't add any new feature atm but is used to precompile some common
# shooting workflows

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using BoundaryValueDiffEq, OrdinaryDiffEq
end

@setup_workload begin
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end
    f1(u, p, t) = [u[2], 0]

    function bc1!(residual, u, p, t)
        residual[1] = u(0.0)[1] - 5
        residual[2] = u(5.0)[1]
    end
    bc1(u, p, t) = [u(0.0)[1] - 5, u(5.0)[1]]

    bc1_a!(residual, ua, p) = (residual[1] = ua[1] - 5)
    bc1_b!(residual, ub, p) = (residual[1] = ub[1])

    bc1_a(ua, p) = [ua[1] - 5]
    bc1_b(ub, p) = [ub[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [
        BVProblem(f1!, bc1!, u0, tspan),
        BVProblem(f1, bc1, u0, tspan),
        TwoPointBVProblem(f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype),
        TwoPointBVProblem(f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype),
    ]

    @compile_workload begin
        for prob in probs, alg in (Shooting(Tsit5()), MultipleShooting(10, Tsit5()))
            solve(prob, alg)
        end
    end
end

end
