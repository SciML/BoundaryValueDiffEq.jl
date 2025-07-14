module BoundaryValueDiffEqMIRKN

using ADTypes: ADTypes, AutoSparse, AutoForwardDiff
using ArrayInterface: fast_scalar_indexing
using BandedMatrices: BandedMatrix, Ones
using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqAlgorithm,
                               AbstractBoundaryValueDiffEqCache, BVPJacobianAlgorithm,
                               recursive_flatten, recursive_flatten!, recursive_unflatten!,
                               __concrete_solve_algorithm, diff!, EvalSol, eval_bc_residual,
                               eval_bc_residual!, get_tmp, __maybe_matmul!,
                               __extract_problem_details, __initial_guess,
                               __maybe_allocate_diffcache, __restructure_sol,
                               __get_bcresid_prototype, safe_similar, __vec, __vec_f,
                               __vec_f!, __vec_bc, __vec_bc!, __vec_so_bc!, __vec_so_bc,
                               recursive_flatten_twopoint!, __internal_nlsolve_problem,
                               __extract_mesh, __extract_u0, __has_initial_guess,
                               __initial_guess_length, __initial_guess_on_mesh,
                               __flatten_initial_guess, __build_solution, __Fix3,
                               __default_sparse_ad, __default_nonsparse_ad, get_dense_ad,
                               concrete_jacobian_algorithm, __default_coloring_algorithm,
                               __default_sparsity_detector, interval, __split_kwargs,
                               NoErrorControl

using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using DifferentiationInterface: DifferentiationInterface, Constant, prepare_jacobian
using FastAlmostBandedMatrices: AlmostBandedMatrix, fillpart, exclusive_bandpart,
                                finish_part_setindex!
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize, Dual
using LinearAlgebra
using PreallocationTools: PreallocationTools, DiffCache
using PrecompileTools: @compile_workload, @setup_workload
using Preferences: Preferences
using RecursiveArrayTools: AbstractVectorOfArray, VectorOfArray, recursivecopy,
                           ArrayPartition
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, AbstractBVProblem,
                 StandardSecondOrderBVProblem, StandardBVProblem, __solve, _unwrap_val
using Setfield: @set!, @set
using SparseArrays: sparse

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
