module BoundaryValueDiffEqAscher

using ADTypes: ADTypes, AutoSparse, AutoForwardDiff
using AlmostBlockDiagonals: AlmostBlockDiagonals, IntermediateAlmostBlockDiagonal

using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqAlgorithm,
    AbstractBoundaryValueDiffEqCache, BVPJacobianAlgorithm,
    __extract_problem_details, concrete_jacobian_algorithm,
    __Fix3, __concrete_solve_algorithm,
    __internal_nlsolve_problem, __vec, __vec_f, __vec_f!,
    __vec_bc, __vec_bc!, __extract_mesh, get_dense_ad,
    __get_bcresid_prototype, __split_kwargs, __concrete_kwargs,
    __default_nonsparse_ad, __construct_internal_problem,
    __internal_solve, __build_cost

using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface, Constant, prepare_jacobian
using FastClosures: @closure
using ForwardDiff: ForwardDiff, Dual
using LinearAlgebra
using PreallocationTools: PreallocationTools, DiffCache
using RecursiveArrayTools: VectorOfArray, recursivecopy
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, StandardBVProblem, __solve,
    _unwrap_val
using Setfield: @set!

const DI = DifferentiationInterface

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
