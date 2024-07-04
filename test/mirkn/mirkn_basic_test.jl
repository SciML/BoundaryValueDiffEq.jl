using BoundaryValueDiffEq

for order in (2, 4, 6)
    s = Symbol("MIRK$(order)")
    @eval mirkn_solver(::Val{$order}, args...; kwargs...) = $(s)(args...; kwargs...)
end

function test!(ddu, du, u, p, t)
    ϵ = 0.1
    ddu[1] = u[2]
    ddu[2] = (-u[1]*du[2] - u[3]*du[3])/ϵ
    ddu[3] = (du[1]*u[3] - u[1]*du[3])/ϵ
end
function test(du, u, p, t)
    ϵ = 0.1
    return [u[2], (-u[1]*du[2] - u[3]*du[3])/ϵ,  (du[1]*u[3] - u[1]*du[3])/ϵ]
end

function bc!(res, du, u, p, t)
    res[1] = u[1][1]
    res[2] = u[end][1]
    res[3] = u[1][3] + 1
    res[4] = u[end][3] - 1
    res[5] = du[1][1]
    res[6] = du[end][1]
end

function bc(du, u, p, t)
    return [u[1][1], u[end][1], u[1][3] + 1, u[end][3] - 1, du[1][1], du[end][1]]
end
function bca!(resa, du, u, p)
    resa[1]=u[1]
    resa[2]=u[3] + 1
    resa[3]=du[1]
end
function bcb!(resb, du, u, p)
    resb[1]=u[1]
    resb[2]=u[3] - 1
    resb[3]=du[1]
end

function bca(du, u, p)
    [u[1], u[3] + 1, du[1]]
end
function bcb(du, u, p)
    [u[1], u[3] - 1, du[1]]
end

u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 1.0)

probArr = [SecondOrderBVProblem(test!, bc!, u0, tspan),
    SecondOrderBVProblem(test, bc, u0, tspan),
    TwoPointSecondOrderBVProblem(test!, (bca!, bcb!), u0, tspan),
    TwoPointSecondOrderBVProblem(test, (bca, bcb), u0, tspan)]

@testset "MIRKN$order" for order in (2, 4, 6)
    @testset "Problem $i" for i in 1:4
        sol = solve(probArr[i], mirkn_solver(Val(order)); dt = 0.01)
        @test SciMLBase.successful_retcode(sol)
    end
end