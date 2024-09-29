module BoundaryValueDiffEqShooting

import PrecompileTools: @compile_workload, @setup_workload

using ADTypes, Adapt, ArrayInterface, DiffEqBase, ForwardDiff, LinearAlgebra,
      NonlinearSolve, Preferences, RecursiveArrayTools, Reexport, SciMLBase, Setfield,
      SparseDiffTools

using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using BandedMatrices, FastAlmostBandedMatrices, SparseArrays

import BoundaryValueDiffEq: BoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
                            recursive_flatten, recursive_flatten!, recursive_unflatten!,
                            __concrete_nonlinearsolve_algorithm, diff!, __any_sparse_ad,
                            __FastShortcutBVPCompatibleNonlinearPolyalg,
                            __FastShortcutBVPCompatibleNLLSPolyalg, __cache_trait,
                            concrete_jacobian_algorithm, eval_bc_residual,
                            eval_bc_residual!, get_tmp, __maybe_matmul!, __append_similar!,
                            __extract_problem_details, __initial_guess,
                            __default_nonsparse_ad, __maybe_allocate_diffcache,
                            __get_bcresid_prototype, __similar, __vec, __vec_f, __vec_f!,
                            __vec_bc, __vec_bc!, __materialize_jacobian_algorithm,
                            recursive_flatten_twopoint!, __unsafe_nonlinearfunction,
                            __internal_nlsolve_problem, NoDiffCacheNeeded, DiffCacheNeeded,
                            __extract_mesh, __extract_u0, __has_initial_guess,
                            __initial_guess_length, __initial_guess_on_mesh,
                            __flatten_initial_guess, __get_non_sparse_ad, __build_solution,
                            __Fix3, __sparse_jacobian_cache, __sparsity_detection_alg,
                            _sparse_like, ColoredMatrix

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
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
include("single_shooting.jl")
include("multiple_shooting.jl")
include("sparse_jacobians.jl")

export Shooting, MultipleShooting
export BVPJacobianAlgorithm

end # module BoundaryValueDiffEqShooting
