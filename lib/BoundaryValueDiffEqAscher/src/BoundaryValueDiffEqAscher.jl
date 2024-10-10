module BoundaryValueDiffEqAscher

using ADTypes
using LinearAlgebra
using AlmostBlockDiagonals
using SciMLBase
using NonlinearSolve
using ForwardDiff
using FastClosures
using RecursiveArrayTools
using Reexport
using Setfield
using ConcreteStructs
using PreallocationTools

import BoundaryValueDiffEq.BVPJacobianAlgorithm
import BoundaryValueDiffEq: __extract_problem_details, concrete_jacobian_algorithm, __Fix3,
                            __concrete_nonlinearsolve_algorithm, __unsafe_nonlinearfunction,
                            BoundaryValueDiffEqAlgorithm, __sparse_jacobian_cache, __vec,
                            __vec_f, __vec_f!, __vec_bc, __vec_bc!

import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val

@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("ascher_tableaus.jl")
include("ascher.jl")
include("adaptivity.jl")
include("collocation.jl")

export Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7

end # module BoundaryValueDiffEqAscher
