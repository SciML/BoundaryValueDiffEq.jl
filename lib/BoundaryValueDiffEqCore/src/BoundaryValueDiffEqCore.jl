module BoundaryValueDiffEqCore

using Adapt: adapt
using ADTypes: ADTypes, AbstractADType, AutoSparse, AutoForwardDiff, AutoFiniteDiff,
    AutoPolyesterForwardDiff
using ArrayInterface: parameterless_type
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using DifferentiationInterface: SecondOrder
using ForwardDiff: ForwardDiff, pickchunksize
using KernelAbstractions: KernelAbstractions, CPU, @index, @kernel, synchronize
using Integrals: Integrals, IntegralProblem
using LinearAlgebra: LinearAlgebra, mul!, UniformScaling
using LinearSolve: KrylovJL_GMRES, KrylovJL_LSMR, UMFPACKFactorization
using LineSearch: BackTracking
using NonlinearSolveFirstOrder: NonlinearSolveFirstOrder,
    GaussNewton, LevenbergMarquardt, NewtonRaphson, TrustRegion
using NonlinearSolveBase: NonlinearSolveBase, NonlinearSolvePolyAlgorithm, NonlinearVerbosity
using OptimizationBase: OptimizationBase, OptimizationVerbosity
using PreallocationTools: PreallocationTools, DiffCache, get_tmp
using RecursiveArrayTools: AbstractVectorOfArray, VectorOfArray, DiffEqArray
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractBVProblem, BVProblem, NonlinearFunction,
    NonlinearLeastSquaresProblem, NonlinearProblem, OptimizationFunction,
    OptimizationProblem, SecondOrderBVProblem, StandardBVProblem,
    StandardSecondOrderBVProblem, TwoPointBVProblem, TwoPointSecondOrderBVProblem,
    __solve
using SciMLLogging: SciMLLogging, Silent,
    InfoLevel, WarnLevel, @verbosity_specifier,
    None, Minimal, Standard, Detailed, All
using SciMLPublic: @public
using Setfield: @set!
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, nnz, rowvals, sparse
using SparseConnectivityTracer: SparseConnectivityTracer, TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using SciMLStructures: SciMLStructures

@reexport using NonlinearSolveFirstOrder:
    GaussNewton, LevenbergMarquardt, NewtonRaphson, TrustRegion
@reexport using SciMLBase:
    BVPFunction, BVProblem, DynamicalBVPFunction, NonlinearFunction,
    NonlinearLeastSquaresProblem, NonlinearProblem, OptimizationFunction,
    OptimizationProblem, ReturnCode, SecondOrderBVProblem, TwoPointBVProblem,
    TwoPointSecondOrderBVProblem, init, remake, solve, successful_retcode

include("verbosity.jl")
include("types.jl")
include("solution_utils.jl")
include("utils.jl")
include("device_sparse.jl")
include("device_linsolve.jl")
include("internal_problems.jl")
include("algorithms.jl")
include("abstract_types.jl")
include("alg_utils.jl")
include("default_internal_solve.jl")
include("calc_errors.jl")

function SciMLBase.__solve(
        prob::AbstractBVProblem,
        alg::AbstractBoundaryValueDiffEqAlgorithm, args...; kwargs...
    )
    cache = SciMLBase.__init(prob, alg, args...; kwargs...)
    return SciMLBase.solve!(cache)
end

export AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm
export DefectControl, GlobalErrorControl, SequentialErrorControl, HybridErrorControl,
    NoErrorControl
export HOErrorControl, REErrorControl
export integral
export BVPVerbosity, _process_verbose_param, DEFAULT_VERBOSE

# Internal API consumed by the solver sublibraries (BoundaryValueDiffEqMIRK,
# BoundaryValueDiffEqFIRK, BoundaryValueDiffEqShooting, BoundaryValueDiffEqAscher,
# BoundaryValueDiffEqMIRKN). Marked public so the sublibraries can import these
# without ExplicitImports flagging them; not exported because they are not part
# of the user-facing API.
@public __device_nlls_linsolve, __device_square_linsolve
@public __default_linsolve, __default_sparse_linsolve,
    __concrete_device_solve_algorithm, __needs_sparse_damping

@public __device_boundary_pattern, __device_sparse_structure,
    __prepare_device_jacobian, __bvp_device_sparse_group

@public __device_parameter, __device_copy_parameter!, __device_host_parameter,
    __device_reshape, __device_singular!, __device_eval!, __device_initial_state,
    __device_initial_backend, __device_bc_sizes, __device_validate_ad, __device_function

@public SparseJacobianCache,
    BVPTunableRHS,
    __bvp_device_ad_jacobian!,
    __device_jacobian!,
    __device_jacobian_products,
    __device_residual!,
    __bvp_device_unknowns, __bvp_device_residual_prototype, __bvp_device_jacobian_plan

@public AbstractBoundaryValueDiffEqCache, AbstractErrorControl, DiffCacheNeeded,
    __device_sparse_linsolve,
    EvalSol, NoDiffCacheNeeded, __FastShortcutNonlinearPolyalg, __Fix3,
    __add_singular_term!, __any_sparse_ad, __build_cost, __build_solution,
    __cache_trait, __concrete_kwargs, __concrete_solve_algorithm,
    __construct_internal_problem, __default_coloring_algorithm,
    __default_nonsparse_ad, __default_sparse_ad, __default_sparsity_detector,
    __device_sparse_matrix, __device_sparse_supported,
    __extract_mesh, __extract_problem_details, __extract_u0,
    __flatten_initial_guess, __get_bcresid_prototype, __get_non_sparse_ad,
    __has_initial_guess, __initial_guess, __initial_guess_length,
    __initial_guess_on_mesh, __internal_nlsolve_problem,
    __internal_optimization_problem, __internal_solve,
    __materialize_jacobian_algorithm, __maybe_allocate_diffcache, __maybe_matmul!,
    __needs_diffcache, __resize!, __restructure_sol, __split_kwargs,
    __tunable_part, __use_both_error_control, __vec, __vec_bc, __vec_bc!,
    __vec_f, __vec_f!, __vec_so_bc, __vec_so_bc!, _sparse_like,
    __apply_mass_matrix!, __get_algebraic_indices, __mass_stage_entry,
    __mass_mesh_entry, __subtract_mass_stage!, __apply_algebraic_constraint!,
    __is_algebraic, __check_dae_adaptivity,
    concrete_jacobian_algorithm, diff!, eval_bc_residual, eval_bc_residual!,
    get_dense_ad, interval, nodual_value, recursive_flatten, recursive_flatten!,
    recursive_flatten_twopoint!, recursive_unflatten!, safe_similar, _unwrap_val

end
