using Base.Test
using BoundaryValueDiffEq, DiffEqBase, DiffEqDevTools

# First order test
function func_1!(x, y, du)
    du[1] = y[2]
    du[2] = 0
end

# Not able to change the initial condition.
# Hard coded solution.
func_1!(::Type{Val{:analytic}}, x, y0) = [5-x,-1]

function boundary!(residual, ua)
    residual[1] = ua[1][1]-5
    residual[2] = ua[end][1]
end

# Second order linear test
function func_2!(x, y, du)
    du[1] = y[2]
    du[2] = -y[1]
end

# Not able to change the initial condition.
# Hard coded solution.
func_2!(::Type{Val{:analytic}}, x, y) = [5*(cos(x) - cot(5)*sin(x)),
                                         5*(-cos(x)*Cot(5) - sin(x))]

tspan = (0.,5.)
u0 = [5.,-3.5]
probArr = [BVProblem(func_1!, boundary!, u0, tspan),
           BVProblem(func_2!, boundary!, u0, tspan)]
testTol = 0.2

println("Convergence Test on Linear")
dts = 1.//2.^(8:-1:4)

solve(probArr[1], MIRK(4))
for i = 1:2
    println("MIRK")
    prob = probArr[i]
    sim = test_convergence(dts,prob,MIRK(4))
    @test abs(sim.𝒪est[:final]-1) < testTol
end
