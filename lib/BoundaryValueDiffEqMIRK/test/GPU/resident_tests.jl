using GPUArrays, JLArrays, LinearSolve
using Test

include("resident_backend_tests.jl")

# JLBackend is synchronous but may not define synchronize.
if !hasmethod(BoundaryValueDiffEqMIRK.KernelAbstractions.synchronize, Tuple{JLBackend})
    BoundaryValueDiffEqMIRK.KernelAbstractions.synchronize(::JLBackend) = nothing
end

@testset "Resident MIRK with scalar indexing disabled" begin
    GPUArrays.allowscalar(false)
    # JLArrays has no vendor LU.
    nlsolve = NewtonRaphson(; linsolve = KrylovJL_GMRES(), concrete_jac = true)
    nllssolve = GaussNewton(; linsolve = KrylovJL_LSMR(), concrete_jac = true)
    test_resident_backend(
        JLArray, u -> u isa JLArray, JLBackend(); algorithms = (MIRK4,), nlsolve, nllssolve
    )
end
