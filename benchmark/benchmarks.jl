using BenchmarkTools
using BoundaryValueDiffEq, OrdinaryDiffEq, NonlinearSolveFirstOrder

include("simple_pendulum.jl")

function create_benchmark()
    suite = BenchmarkGroup()
    suite["Simple Pendulum"] = create_simple_pendulum_benchmark()
    return suite
end

const SUITE = create_benchmark()
