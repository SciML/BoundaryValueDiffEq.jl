module BoundaryValueDiffEqShooting

using ADTypes
using ArrayInterface: fast_scalar_indexing
using BandedMatrices: BandedMatrix, Ones
using BoundaryValueDiffEqCore: AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
                               recursive_flatten, recursive_flatten!, recursive_unflatten!,
                               __concrete_nonlinearsolve_algorithm, diff!, __any_sparse_ad,
                               __cache_trait, concrete_jacobian_algorithm, eval_bc_residual,
                               eval_bc_residual!, get_tmp, __maybe_matmul!,
                               __extract_problem_details, __initial_guess,
                               __default_coloring_algorithm, __default_sparsity_detector,
                               __maybe_allocate_diffcache, __get_bcresid_prototype,
                               safe_similar, __vec, __vec_f, __vec_f!, __vec_bc, __vec_bc!,
                               __materialize_jacobian_algorithm, __default_nonsparse_ad,
                               recursive_flatten_twopoint!, __internal_nlsolve_problem,
                               NoDiffCacheNeeded, DiffCacheNeeded, __extract_mesh,
                               __extract_u0, __has_initial_guess, __initial_guess_length,
                               __initial_guess_on_mesh, __flatten_initial_guess,
                               __get_non_sparse_ad, __build_solution, __Fix3, get_dense_ad

using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using DifferentiationInterface: DifferentiationInterface, Constant, prepare_jacobian,
                                overloaded_input_type
using FastAlmostBandedMatrices: AlmostBandedMatrix, fillpart, exclusive_bandpart,
                                finish_part_setindex!
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using LinearAlgebra
using PrecompileTools: @compile_workload, @setup_workload
using Preferences: Preferences
using Reexport: @reexport
using RecursiveArrayTools: ArrayPartition, DiffEqArray, VectorOfArray
using SciMLBase: SciMLBase, AbstractDiffEqInterpolation, StandardBVProblem, __solve,
                 _unwrap_val
using Setfield: @set!, @set
using SparseArrays: sparse

const DI = DifferentiationInterface

@reexport using ADTypes, BoundaryValueDiffEqCore, SciMLBase

include("algorithms.jl")
include("single_shooting.jl")
include("multiple_shooting.jl")
include("sparse_jacobians.jl")

export Shooting, MultipleShooting

end # module BoundaryValueDiffEqShooting
