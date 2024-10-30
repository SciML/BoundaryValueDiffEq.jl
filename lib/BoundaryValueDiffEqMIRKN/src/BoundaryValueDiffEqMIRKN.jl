module BoundaryValueDiffEqMIRKN

import PrecompileTools: @compile_workload, @setup_workload

using ADTypes, Adapt, ArrayInterface, BoundaryValueDiffEqCore, DiffEqBase, ForwardDiff,
      LinearAlgebra, NonlinearSolve, Preferences, RecursiveArrayTools, Reexport, SciMLBase,
      Setfield, SparseDiffTools

using PreallocationTools: PreallocationTools, DiffCache

# Special Matrix Types
using BandedMatrices, FastAlmostBandedMatrices, SparseArrays

import BoundaryValueDiffEqCore: BoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
                                recursive_flatten, recursive_flatten!, recursive_unflatten!,
                                __concrete_nonlinearsolve_algorithm, diff!,
                                __FastShortcutBVPCompatibleNonlinearPolyalg,
                                __FastShortcutBVPCompatibleNLLSPolyalg, eval_bc_residual,
                                eval_bc_residual!, get_tmp, __maybe_matmul!,
                                __append_similar!, __extract_problem_details,
                                __initial_guess, __maybe_allocate_diffcache,
                                __get_bcresid_prototype, __similar, __vec, __vec_f,
                                __vec_f!, __vec_bc, __vec_bc!, recursive_flatten_twopoint!,
                                __unsafe_nonlinearfunction, __internal_nlsolve_problem,
                                __extract_mesh, __extract_u0, __has_initial_guess,
                                __initial_guess_length, __initial_guess_on_mesh,
                                __flatten_initial_guess, __build_solution, __Fix3,
                                __sparse_jacobian_cache, __sparsity_detection_alg,
                                _sparse_like, ColoredMatrix, __default_sparse_ad,
                                __default_nonsparse_ad

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors, parameterless_type, undefmatrix, fast_scalar_indexing
import ConcreteStructs: @concrete
import DiffEqBase: solve
import FastClosures: @closure
import ForwardDiff: ForwardDiff, pickchunksize
import Logging
import RecursiveArrayTools: ArrayPartition, DiffEqArray
import SciMLBase: AbstractDiffEqInterpolation, AbstractBVProblem,
                  StandardSecondOrderBVProblem, StandardBVProblem, __solve, _unwrap_val

@reexport using ADTypes, DiffEqBase, NonlinearSolve, SparseDiffTools, SciMLBase

include("utils.jl")
include("types.jl")
include("algorithms.jl")
include("mirkn.jl")
include("alg_utils.jl")
include("collocation.jl")
include("mirkn_tableaus.jl")

function __solve(
        prob::AbstractBVProblem, alg::BoundaryValueDiffEqAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

export MIRKN4, MIRKN6
export BVPJacobianAlgorithm

end
