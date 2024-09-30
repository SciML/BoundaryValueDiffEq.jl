module BoundaryValueDiffEqCore

using ADTypes, Adapt, ArrayInterface, DiffEqBase, ForwardDiff, LinearAlgebra,
      RecursiveArrayTools, Reexport, SciMLBase, Setfield, SparseDiffTools

using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using SparseArrays

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, fast_scalar_indexing
import ConcreteStructs: @concrete
import DiffEqBase: solve
import ForwardDiff: ForwardDiff, pickchunksize
import Logging
import RecursiveArrayTools: VectorOfArray, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val

@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("default_nlsolve.jl")
include("sparse_jacobians.jl")

function __solve(prob::BVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export BVPJacobianAlgorithm

end
