module BoundaryValueDiffEqAscher

using ADTypes: ADTypes, AutoSparse, AutoForwardDiff
using AlmostBlockDiagonals: AlmostBlockDiagonals, IntermediateAlmostBlockDiagonal
using BoundaryValueDiffEqCore: BVPJacobianAlgorithm, __extract_problem_details,
                               concrete_jacobian_algorithm, __Fix3,
                               __concrete_nonlinearsolve_algorithm,
                               __internal_nlsolve_problem, BoundaryValueDiffEqAlgorithm,
                               __vec, __vec_f, __vec_f!, __vec_bc, __vec_bc!,
                               __extract_mesh, get_dense_ad, __sparse_jacobian_cache
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using FastClosures: @closure
using ForwardDiff: ForwardDiff, Dual
using LinearAlgebra
using PreallocationTools: PreallocationTools, DiffCache
using RecursiveArrayTools: VectorOfArray, recursivecopy
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, StandardBVProblem, __solve,
                 _unwrap_val
using Setfield: @set!
using SparseDiffTools: init_jacobian, sparse_jacobian, sparse_jacobian_cache,
                       sparse_jacobian!, SymbolicsSparsityDetection, NoSparsityDetection

@reexport using ADTypes, BoundaryValueDiffEqCore, SciMLBase

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
