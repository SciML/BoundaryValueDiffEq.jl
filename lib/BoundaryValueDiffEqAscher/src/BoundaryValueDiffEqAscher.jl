module BoundaryValueDiffEqAscher

using ADTypes: ADTypes, AutoSparse, AutoForwardDiff
using AlmostBlockDiagonals: AlmostBlockDiagonals, IntermediateAlmostBlockDiagonal

using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore,
    AbstractBoundaryValueDiffEqAlgorithm,
    AbstractBoundaryValueDiffEqCache, BVPJacobianAlgorithm,
    DEFAULT_VERBOSE, GlobalErrorControl, _process_verbose_param,
    __extract_problem_details, concrete_jacobian_algorithm,
    __concrete_solve_algorithm,
    __vec, __vec_f, __vec_f!,
    __vec_bc, __vec_bc!, __extract_mesh, get_dense_ad,
    __get_bcresid_prototype, __split_kwargs, __concrete_kwargs,
    __default_nonsparse_ad, __construct_internal_problem,
    __internal_solve, __build_cost

using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, I, norm, rank
using Reexport: @reexport
using SciMLBase: SciMLBase, BVProblem, ReturnCode, StandardBVProblem,
    TwoPointBVProblem, isinplace, solve
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
