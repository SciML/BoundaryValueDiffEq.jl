module BoundaryValueDiffEqAscher

using ADTypes: ADTypes, AutoSparse
using AlmostBlockDiagonals: AlmostBlockDiagonals, IntermediateAlmostBlockDiagonal

using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore,
    AbstractBoundaryValueDiffEqAlgorithm,
    AbstractBoundaryValueDiffEqCache, BVPJacobianAlgorithm,
    DEFAULT_VERBOSE, GlobalErrorControl, _process_verbose_param,
    __extract_problem_details, concrete_jacobian_algorithm,
    __concrete_solve_algorithm,
    __vec, __vec_f, __vec_f!,
    __vec_bc, __vec_bc!, __extract_mesh, get_dense_ad,
    __get_bcresid_prototype, __split_kwargs, __concrete_kwargs,
    __default_nonsparse_ad, __construct_internal_problem,
    __internal_solve, __build_cost

using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface, Constant
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, I, norm, rank
using SciMLBase: SciMLBase, BVProblem, ReturnCode, StandardBVProblem,
    TwoPointBVProblem, isinplace, solve

# The public API that BoundaryValueDiffEqAscher reexports (see the second `export` block
# below), so that `using BoundaryValueDiffEqAscher` on its own is enough to pick an AD
# backend, build a `BVProblem` or `TwoPointBVProblem` (including the semi-explicit BVDAE
# form carried by a `BVPFunction` mass matrix), configure the solve, run it, and inspect
# the result. Every name stays owned and documented by ADTypes, BoundaryValueDiffEqCore,
# NonlinearSolveFirstOrder or SciMLBase; the set is documented on the Reexported API docs
# page and approved via `reexports_allow` in test/qa/qa.jl.
using ADTypes: AutoEnzyme, AutoFiniteDiff, AutoForwardDiff, AutoMooncake,
    AutoPolyesterForwardDiff
using BoundaryValueDiffEqCore: BVPVerbosity, DefectControl, GaussNewton, HOErrorControl,
    HybridErrorControl, LevenbergMarquardt, NewtonRaphson, NoErrorControl, REErrorControl,
    SequentialErrorControl, TrustRegion, integral
using SciMLBase: BVPFunction, init, remake, solve!, successful_retcode

using Setfield: @set!

const DI = DifferentiationInterface

include("types.jl")
include("utils.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("ascher_tableaus.jl")
include("ascher.jl")
include("adaptivity.jl")
include("collocation.jl")

export Ascher1, Ascher2, Ascher3, Ascher4, Ascher5, Ascher6, Ascher7

# Reexported ADTypes / BoundaryValueDiffEqCore / NonlinearSolveFirstOrder / SciMLBase
# API; approved via `reexports_allow` in test/qa/qa.jl.
export AutoEnzyme, AutoFiniteDiff, AutoForwardDiff, AutoMooncake, AutoPolyesterForwardDiff,
    AutoSparse
export BVPJacobianAlgorithm, BVPVerbosity, DEFAULT_VERBOSE
export DefectControl, GlobalErrorControl, SequentialErrorControl, HybridErrorControl,
    NoErrorControl
export HOErrorControl, REErrorControl
export integral
export GaussNewton, LevenbergMarquardt, NewtonRaphson, TrustRegion
export BVPFunction, BVProblem, ReturnCode, TwoPointBVProblem, init, remake, solve, solve!,
    successful_retcode

end # module BoundaryValueDiffEqAscher
