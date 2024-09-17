module BoundaryValueDiffEqShooting

import PrecompileTools: @compile_workload, @setup_workload

using ADTypes, Adapt, ArrayInterface, DiffEqBase, ForwardDiff, LinearAlgebra,
      NonlinearSolve, OrdinaryDiffEq, Preferences, RecursiveArrayTools, Reexport, SciMLBase,
      Setfield, SparseDiffTools

using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using BandedMatrices, FastAlmostBandedMatrices, SparseArrays

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
import BoundaryValueDiffEq: BVPJacobianAlgorithm, recursive_flatten, recursive_flatten!, recursive_unflatten!,
    BoundaryValueDiffEqAlgorithm, __concrete_nonlinearsolve_algorithm, __split_mirk_kwargs, __cache_trait, __get_non_sparse_ad,
    __FastShortcutBVPCompatibleNonlinearPolyalg, __FastShortcutBVPCompatibleNLLSPolyalg, __materialize_jacobian_algorithm,
    concrete_jacobian_algorithm, eval_bc_residual, eval_bc_residual!, get_tmp, __maybe_matmul!, __default_nonsparse_ad,
    __append_similar!, __extract_problem_details, __initial_guess, __maybe_allocate_diffcache, DiffCacheNeeded, NoDiffCacheNeeded,
    __get_bcresid_prototype, __similar, __vec, __vec_f, __vec_f!, __vec_bc, __vec_bc!, __perform_mirk_iteration,
    __unsafe_nonlinearfunction, __internal_nlsolve_problem, __generate_sparse_jacobian_prototype, __any_sparse_ad,
    __extract_mesh, __extract_u0, __has_initial_guess, __initial_guess_length, __initial_guess_on_mesh,
    __flatten_initial_guess, __build_solution, __Fix3, __sparse_jacobian_cache, __sparsity_detection_alg, _sparse_like, ColoredMatrix

import ConcreteStructs: @concrete
import DiffEqBase: solve
import FastClosures: @closure
import ForwardDiff: ForwardDiff, pickchunksize
import Logging
import RecursiveArrayTools: ArrayPartition, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation, StandardBVProblem, __solve, _unwrap_val

@reexport using ADTypes, DiffEqBase, NonlinearSolve, OrdinaryDiffEq, SparseDiffTools,
                SciMLBase

include("algorithms.jl")
include("sparse_jacobians.jl")
include("single_shooting.jl")
include("multiple_shooting.jl")

export Shooting, MultipleShooting

end # module BoundaryValueDiffEqShooting
