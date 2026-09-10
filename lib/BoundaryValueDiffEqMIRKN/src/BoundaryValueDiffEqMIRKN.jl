module BoundaryValueDiffEqMIRKN

using ADTypes: ADTypes, AutoSparse
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
using SciMLBase: SciMLBase, ReturnCode, SecondOrderBVProblem,
    StandardSecondOrderBVProblem, TwoPointSecondOrderBVProblem, isinplace, remake

# The public API that BoundaryValueDiffEqMIRKN reexports (see the second `export` block
# below), so that `using BoundaryValueDiffEqMIRKN` on its own is enough to pick an AD
# backend, build a `SecondOrderBVProblem` or `TwoPointSecondOrderBVProblem`, configure the
# solve, run it, and inspect the result. Every name stays owned and documented by ADTypes,
# BoundaryValueDiffEqCore, NonlinearSolveFirstOrder or SciMLBase; the set is documented on
# the Reexported API docs page and approved via `reexports_allow` in test/qa/qa.jl.
using ADTypes: AutoEnzyme, AutoFiniteDiff, AutoForwardDiff, AutoMooncake,
    AutoPolyesterForwardDiff
using BoundaryValueDiffEqCore: BVPVerbosity, GaussNewton, LevenbergMarquardt,
    NewtonRaphson, TrustRegion, integral
using SciMLBase: DynamicalBVPFunction, init, solve, solve!, successful_retcode

using Setfield: @set!

const DI = DifferentiationInterface

include("types.jl")
include("algorithms.jl")
include("mirkn.jl")
include("alg_utils.jl")
include("collocation.jl")
include("mirkn_tableaus.jl")
include("interpolation.jl")

export MIRKN4, MIRKN6

# Reexported ADTypes / BoundaryValueDiffEqCore / NonlinearSolveFirstOrder / SciMLBase
# API; approved via `reexports_allow` in test/qa/qa.jl.
export AutoEnzyme, AutoFiniteDiff, AutoForwardDiff, AutoMooncake, AutoPolyesterForwardDiff,
    AutoSparse
export BVPJacobianAlgorithm, BVPVerbosity, DEFAULT_VERBOSE, NoErrorControl
export integral
export GaussNewton, LevenbergMarquardt, NewtonRaphson, TrustRegion
export DynamicalBVPFunction, ReturnCode, SecondOrderBVProblem,
    TwoPointSecondOrderBVProblem, init, remake, solve, solve!, successful_retcode

end
