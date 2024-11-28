module BoundaryValueDiffEqMIRKN

using ADTypes: ADTypes, AutoSparse, AutoForwardDiff
using ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
using BandedMatrices: BandedMatrix, Ones
using BoundaryValueDiffEqCore: BoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
                               recursive_flatten, recursive_flatten!, recursive_unflatten!,
                               __concrete_nonlinearsolve_algorithm, diff!,
                               __FastShortcutBVPCompatibleNonlinearPolyalg,
                               __FastShortcutBVPCompatibleNLLSPolyalg, eval_bc_residual,
                               eval_bc_residual!, get_tmp, __maybe_matmul!,
                               __append_similar!, __extract_problem_details,
                               __initial_guess, __maybe_allocate_diffcache,
                               __get_bcresid_prototype, __similar, __vec, __vec_f, __vec_f!,
                               __vec_bc, __vec_bc!, __vec_so_bc!, __vec_so_bc,
                               recursive_flatten_twopoint!, __internal_nlsolve_problem,
                               __extract_mesh, __extract_u0, __has_initial_guess,
                               __initial_guess_length, __initial_guess_on_mesh,
                               __flatten_initial_guess, __build_solution, __Fix3,
                               _sparse_like, __default_sparse_ad, __default_nonsparse_ad,
                               get_dense_ad

using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using DifferentiationInterface: DifferentiationInterface, Constant
using FastAlmostBandedMatrices: AlmostBandedMatrix, fillpart, exclusive_bandpart,
                                finish_part_setindex!
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using LinearAlgebra
using PreallocationTools: PreallocationTools, DiffCache
using PrecompileTools: @compile_workload, @setup_workload
using Preferences: Preferences
using RecursiveArrayTools: VectorOfArray, recursivecopy, ArrayPartition
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, AbstractBVProblem,
                 StandardSecondOrderBVProblem, StandardBVProblem, __solve, _unwrap_val
using Setfield: @set!, @set
using SparseArrays: sparse
using SparseConnectivityTracer: SparseConnectivityTracer
using SparseMatrixColorings: ColoringProblem, GreedyColoringAlgorithm,
                             ConstantColoringAlgorithm, row_colors, column_colors, coloring,
                             LargestFirst

const DI = DifferentiationInterface

@reexport using ADTypes, BoundaryValueDiffEqCore, SciMLBase

include("utils.jl")
include("types.jl")
include("algorithms.jl")
include("mirkn.jl")
include("alg_utils.jl")
include("collocation.jl")
include("mirkn_tableaus.jl")

export MIRKN4, MIRKN6
export BVPJacobianAlgorithm

end
