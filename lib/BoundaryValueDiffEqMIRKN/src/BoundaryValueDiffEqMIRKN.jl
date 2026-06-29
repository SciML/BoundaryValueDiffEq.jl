module BoundaryValueDiffEqMIRKN

using ADTypes: ADTypes, AutoSparse, AutoForwardDiff
using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore,
    AbstractBoundaryValueDiffEqAlgorithm,
    AbstractBoundaryValueDiffEqCache, BVPJacobianAlgorithm,
    DEFAULT_VERBOSE, _process_verbose_param,
    recursive_flatten!, recursive_unflatten!,
    __concrete_solve_algorithm, EvalSol, eval_bc_residual,
    eval_bc_residual!, __maybe_matmul!,
    __extract_problem_details,
    __maybe_allocate_diffcache, __restructure_sol,
    safe_similar, __vec_f,
    __vec_f!, __vec_so_bc!, __vec_so_bc,
    __extract_mesh,
    __initial_guess_on_mesh,
    __build_solution,
    get_dense_ad,
    concrete_jacobian_algorithm, __default_coloring_algorithm,
    __default_sparsity_detector, interval,
    NoErrorControl, __construct_internal_problem,
    __concrete_kwargs, __internal_solve

using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using LinearAlgebra: LinearAlgebra
using PreallocationTools: PreallocationTools, get_tmp
using Preferences: Preferences
using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition
using Reexport: @reexport
using SciMLBase: SciMLBase, ReturnCode, SecondOrderBVProblem,
    StandardSecondOrderBVProblem, TwoPointSecondOrderBVProblem, isinplace, remake
using Setfield: @set!

const DI = DifferentiationInterface

@reexport using ADTypes, BoundaryValueDiffEqCore, SciMLBase

include("types.jl")
include("algorithms.jl")
include("mirkn.jl")
include("alg_utils.jl")
include("collocation.jl")
include("mirkn_tableaus.jl")
include("interpolation.jl")

export MIRKN4, MIRKN6

end
