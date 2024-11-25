module BoundaryValueDiffEqMIRKN

using PrecompileTools: @compile_workload, @setup_workload
using SparseMatrixColorings: ColoringProblem, GreedyColoringAlgorithm,
                             ConstantColoringAlgorithm, row_colors, column_colors, coloring,
                             LargestFirst
using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using BandedMatrices: BandedMatrix, Ones
using SparseArrays
using LinearAlgebra
using FastAlmostBandedMatrices: AlmostBandedMatrix, fillpart, exclusive_bandpart,
                                finish_part_setindex!

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
                               __sparse_jacobian_cache, __sparsity_detection_alg,
                               _sparse_like, ColoredMatrix, __default_sparse_ad,
                               __default_nonsparse_ad, get_dense_ad

using ADTypes: ADTypes
using ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using Logging
using RecursiveArrayTools: VectorOfArray, recursivecopy, ArrayPartition
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, AbstractBVProblem,
                 StandardSecondOrderBVProblem, StandardBVProblem, __solve, _unwrap_val
using Setfield: @set!
using SparseConnectivityTracer: SparseConnectivityTracer
using Reexport: @reexport
using Preferences: Preferences

const DI = DifferentiationInterface

@reexport using BoundaryValueDiffEqCore, SciMLBase

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
