module BoundaryValueDiffEqMIRK

using ADTypes
using ArrayInterface: fast_scalar_indexing
using BandedMatrices: BandedMatrix, Ones
using BoundaryValueDiffEqCore: BoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
                               recursive_flatten, recursive_flatten!, recursive_unflatten!,
                               __concrete_nonlinearsolve_algorithm, diff!,
                               __FastShortcutBVPCompatibleNonlinearPolyalg,
                               __FastShortcutBVPCompatibleNLLSPolyalg, EvalSol,
                               concrete_jacobian_algorithm, eval_bc_residual,
                               eval_bc_residual!, get_tmp, __maybe_matmul!, __resize!,
                               __extract_problem_details, __initial_guess,
                               __maybe_allocate_diffcache, __restructure_sol,
                               __get_bcresid_prototype, __similar, __vec, __vec_f, __vec_f!,
                               __vec_bc, __vec_bc!, recursive_flatten_twopoint!,
                               __internal_nlsolve_problem, __extract_mesh, __extract_u0,
                               __has_initial_guess, __initial_guess_length, __cache_trait,
                               __initial_guess_on_mesh, __flatten_initial_guess,
                               __build_solution, __Fix3, get_dense_ad, _sparse_like,
                               AbstractErrorControl, DefectControl, GlobalErrorControl,
                               SequentialErrorControl, HybridErrorControl, HOErrorControl,
                               __use_both_error_control, DiffCacheNeeded, NoDiffCacheNeeded

using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using DifferentiationInterface: DifferentiationInterface, Constant, prepare_jacobian
using FastAlmostBandedMatrices: AlmostBandedMatrix, fillpart, exclusive_bandpart,
                                finish_part_setindex!
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize, Dual
using LinearAlgebra
using RecursiveArrayTools: AbstractVectorOfArray, DiffEqArray, VectorOfArray, recursivecopy,
                           recursivefill!
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, StandardBVProblem, __solve,
                 _unwrap_val
using Setfield: @set!
using Reexport: @reexport
using PreallocationTools: PreallocationTools, DiffCache
using PrecompileTools: @compile_workload, @setup_workload
using Preferences: Preferences
using SparseArrays: sparse

const DI = DifferentiationInterface

@reexport using ADTypes, BoundaryValueDiffEqCore, SciMLBase

include("types.jl")
include("algorithms.jl")
include("mirk.jl")
include("adaptivity.jl")
include("alg_utils.jl")
include("collocation.jl")
include("interpolation.jl")
include("mirk_tableaus.jl")
include("sparse_jacobians.jl")

export MIRK2, MIRK3, MIRK4, MIRK5, MIRK6, MIRK6I
export BVPJacobianAlgorithm

end
