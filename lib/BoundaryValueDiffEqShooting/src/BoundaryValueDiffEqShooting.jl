module BoundaryValueDiffEqShooting

using ADTypes: ADTypes, AutoForwardDiff, AutoSparse
using ArrayInterface: fast_scalar_indexing
using BandedMatrices: BandedMatrix, Ones
using BoundaryValueDiffEqCore: BoundaryValueDiffEqCore,
    AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm,
    recursive_flatten!,
    __concrete_solve_algorithm,
    __cache_trait, concrete_jacobian_algorithm, eval_bc_residual,
    eval_bc_residual!,
    __concrete_kwargs, __extract_problem_details,
    __construct_internal_problem,
    __default_coloring_algorithm,
    __maybe_allocate_diffcache, __get_bcresid_prototype,
    __vec,
    __materialize_jacobian_algorithm, __default_nonsparse_ad,
    NoDiffCacheNeeded, DiffCacheNeeded,
    __extract_u0,
    __initial_guess_on_mesh,
    __get_non_sparse_ad, __build_solution, get_dense_ad,
    __internal_solve, _process_verbose_param, _unwrap_val

using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface,
    overloaded_input_type
using FastClosures: @closure
using ForwardDiff: ForwardDiff, pickchunksize
using LinearAlgebra: LinearAlgebra
using SciMLBase: SciMLBase, BVProblem, EnsembleSerial, EnsembleThreads,
    NonlinearFunction, ODEProblem, StandardBVProblem, TwoPointBVProblem,
    __solve, isinplace, remake, solve, solve!
using SciMLLogging: @SciMLMessage
using Setfield: @set
using SparseArrays: sparse
using OrdinaryDiffEqTsit5: Tsit5
using PrecompileTools: @compile_workload, @setup_workload
using Preferences: Preferences

# The public API that BoundaryValueDiffEqShooting re-exports (see the second `export`
# below), so that `using BoundaryValueDiffEqShooting` on its own is enough to pick an AD
# backend, build a `BVProblem`, choose the inner nonlinear solver, and inspect the
# solution. These names stay owned and documented by ADTypes, BoundaryValueDiffEqCore and
# SciMLBase; the set is the one the previous `@reexport using ADTypes,
# BoundaryValueDiffEqCore, SciMLBase` made public.
using ADTypes: AbstractADType, AbstractColoringAlgorithm, AbstractSparsityDetector,
    AutoChainRules, AutoDiffractor, AutoEnzyme, AutoFastDifferentiation, AutoFiniteDiff,
    AutoFiniteDifferences, AutoGTPSA, AutoHyperHessians, AutoModelingToolkit, AutoMooncake,
    AutoMooncakeForward, AutoPolyesterForwardDiff, AutoReactant, AutoReverseDiff,
    AutoSparseFastDifferentiation, AutoSparseFiniteDiff, AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff, AutoSparseReverseDiff, AutoSparseZygote, AutoSymbolics,
    AutoTapir, AutoTaylorDiff, AutoTracker, AutoZygote, NoAutoDiff, NoAutoDiffSelectedError,
    column_coloring, hessian_sparsity, jacobian_sparsity, row_coloring, symmetric_coloring
using SciMLBase: AbstractAnalyticalProblem, AddVector, AffineOperator, AllObserved,
    AnalyticalProblem, BVPFunction, BatchIntegralFunction, BlockDiagonalOperator,
    CallbackSet, CheckInit, Clocks, ContinuousCallback, ConvexOptimizationProblem,
    DAEFunction, DAEProblem, DAESolution, DDEFunction, DDEProblem, DiagonalOperator,
    DiscreteCallback, DiscreteFunction, DiscreteProblem, DynamicalBVPFunction,
    DynamicalDDEFunction, DynamicalDDEProblem, DynamicalODEFunction, DynamicalODEProblem,
    DynamicalSDEFunction, DynamicalSDEProblem, EigenvalueProblem, EigenvalueSolution,
    EigenvalueTarget, EnsembleAnalysis, EnsembleContext, EnsembleDistributed,
    EnsembleProblem, EnsembleSolution, EnsembleSplitThreads, EnsembleSummary,
    EnsembleTestSolution, FunctionOperator, HomotopyNonlinearFunction, HomotopyProblem,
    IdentityOperator, ImplicitDiscreteFunction, ImplicitDiscreteProblem,
    IncrementingODEFunction, IncrementingODEProblem, IntegralFunction, IntegralProblem,
    IntegralSolution, IntervalNonlinearFunction, IntervalNonlinearProblem,
    InvertibleOperator, LinearAliasSpecifier, LinearProblem, LinearSolution, MatrixOperator,
    MultiObjectiveOptimizationFunction, NoiseProblem, NonlinearLeastSquaresProblem,
    NonlinearProblem, NonlinearSolution, NullOperator, ODEAliasSpecifier, ODEFunction,
    ODEInputFunction, ODESolution, OptimizationFunction, OptimizationProblem,
    OptimizationSolution, PDENoTimeSolution, PDEProblem, PDETimeSeriesSolution,
    RODEFunction, RODEProblem, RODESolution, ReturnCode, SCCNonlinearProblem, SDDEFunction,
    SDDEProblem, SDEFunction, SDEProblem, SampledIntegralProblem, ScalarOperator,
    SciMLOperators, SecondOrderBVProblem, SecondOrderDDEProblem, SecondOrderODEProblem,
    SplitFunction, SplitODEProblem, SplitSDEFunction, SplitSDEProblem, StaticWOperator,
    SteadyStateProblem, SteadyStateSolution, TensorProductOperator, TensorSumOperator,
    TimeDomain, TwoPointBVPFunction, TwoPointDynamicalBVPFunction,
    TwoPointSecondOrderBVProblem, VectorContinuousCallback, WOperator, add_saveat!,
    add_tstop!, addat!, addat_non_user_cache!, addsteps!, auto_dt_reset!, cache_operator,
    change_t_via_interpolation!, check_error, check_keywords, concretize, deleteat!,
    deleteat_non_user_cache!, derivative_discontinuity!, discretize, du_cache, first_tstop,
    full_cache, get_dt, get_du, get_du!, get_proposed_dt, get_rng, get_tmp_cache,
    has_adjoint, has_concretization, has_exp, has_expmv, has_expmv!, has_ldiv, has_ldiv!,
    has_mul, has_mul!, has_rng, has_tstop, init, is_discrete_time_domain, iscached, isclock,
    isconstant, iscontinuous, isconvertible, isdiscrete, islinear, issolverstepclock,
    issquare, kronsum, pop_tstop!, rand_cache, ratenoise_cache,
    reeval_internals_due_to_modification!, reinit!, resize_non_user_cache!, savevalues!,
    set_abstol!, set_proposed_dt!, set_reltol!, set_rng!, set_t!, set_u!, step!,
    supports_solve_rng, symbolic_discretize, terminate!, u_cache, u_modified!,
    update_coefficients, update_coefficients!, user_cache, warn_compat
using BoundaryValueDiffEqCore: AbsNormSafeBestTerminationMode, AbsNormSafeTerminationMode,
    AbsNormTerminationMode, AbsTerminationMode, ArcLengthContinuation, BVPVerbosity,
    DEFAULT_VERBOSE, DampedNewtonDescent, DefectControl, DescentResult, Dogleg,
    EisenstatWalkerForcing2, FastShortcutNLLSPolyalg, GaussNewton,
    GeneralizedFirstOrderAlgorithm, GeodesicAcceleration, GlobalErrorControl,
    HOErrorControl, HomotopyPolyAlgorithm, HomotopySweep, HybridErrorControl,
    KantorovichHomotopy, LevenbergMarquardt, NewtonDescent, NewtonRaphson, NoErrorControl,
    NonlinearSolveBase, NonlinearSolveFirstOrder, NonlinearSolvePolyAlgorithm,
    NonlinearVerbosity, NormTerminationMode, PseudoTransient, REErrorControl,
    RadiusUpdateSchemes, RelNormSafeBestTerminationMode, RelNormSafeTerminationMode,
    RelNormTerminationMode, RelTerminationMode, RobustMultiNewton, SequentialErrorControl,
    SteepestDescent, TraceAll, TraceMinimal, TraceWithJacobianConditionNumber, TrustRegion,
    integral

const DI = DifferentiationInterface

include("algorithms.jl")
include("single_shooting.jl")
include("multiple_shooting.jl")
include("sparse_jacobians.jl")

@setup_workload begin
    function f1!(du, u, p, t)
        du[1] = u[2]
        du[2] = 0
    end
    f1 = (u, p, t) -> [u[2], 0]

    function bc1!(residual, u, p, t)
        residual[1] = u(0.0)[1] - 5
        residual[2] = u(5.0)[1]
    end

    bc1 = (u, p, t) -> [u(0.0)[1] - 5, u(5.0)[1]]

    bc1_a! = (residual, ua, p) -> (residual[1] = ua[1] - 5)
    bc1_b! = (residual, ub, p) -> (residual[1] = ub[1])

    bc1_a = (ua, p) -> [ua[1] - 5]
    bc1_b = (ub, p) -> [ub[1]]

    tspan = (0.0, 5.0)
    u0 = [5.0, -3.5]
    bcresid_prototype = (Array{Float64}(undef, 1), Array{Float64}(undef, 1))

    probs = [
        BVProblem(f1!, bc1!, u0, tspan; nlls = Val(false)),
        BVProblem(f1, bc1, u0, tspan; nlls = Val(false)),
        TwoPointBVProblem(
            f1!, (bc1_a!, bc1_b!), u0, tspan; bcresid_prototype, nlls = Val(false)
        ),
        TwoPointBVProblem(
            f1, (bc1_a, bc1_b), u0, tspan; bcresid_prototype, nlls = Val(false)
        ),
    ]

    jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2))

    algs = []

    if Preferences.@load_preference("PrecompileShooting", true)
        append!(algs, [Shooting(Tsit5(); jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg)
        end
    end

    algs = []

    if Preferences.@load_preference("PrecompileMultipleShooting", true)
        append!(algs, [MultipleShooting(5, Tsit5(); jac_alg)])
    end

    @compile_workload begin
        @sync for prob in probs, alg in algs
            Threads.@spawn solve(prob, alg)
        end
    end
end

export Shooting, MultipleShooting

# Re-exported ADTypes / BoundaryValueDiffEqCore / SciMLBase public API; approved via
# `reexports_allow` in test/qa/qa.jl.
export ADTypes, AbsNormSafeBestTerminationMode, AbsNormSafeTerminationMode,
    AbsNormTerminationMode, AbsTerminationMode, AbstractADType, AbstractAnalyticalProblem,
    AbstractBoundaryValueDiffEqAlgorithm, AbstractColoringAlgorithm,
    AbstractSparsityDetector, AddVector, AffineOperator, AllObserved, AnalyticalProblem,
    ArcLengthContinuation, AutoChainRules, AutoDiffractor, AutoEnzyme,
    AutoFastDifferentiation, AutoFiniteDiff, AutoFiniteDifferences, AutoForwardDiff,
    AutoGTPSA, AutoHyperHessians, AutoModelingToolkit, AutoMooncake, AutoMooncakeForward,
    AutoPolyesterForwardDiff, AutoReactant, AutoReverseDiff, AutoSparse,
    AutoSparseFastDifferentiation, AutoSparseFiniteDiff, AutoSparseForwardDiff,
    AutoSparsePolyesterForwardDiff, AutoSparseReverseDiff, AutoSparseZygote, AutoSymbolics,
    AutoTapir, AutoTaylorDiff, AutoTracker, AutoZygote, BVPFunction, BVPJacobianAlgorithm,
    BVPVerbosity, BVProblem, BatchIntegralFunction, BlockDiagonalOperator,
    BoundaryValueDiffEqCore, CallbackSet, CheckInit, Clocks, ContinuousCallback,
    ConvexOptimizationProblem, DAEFunction, DAEProblem, DAESolution, DDEFunction,
    DDEProblem, DEFAULT_VERBOSE, DampedNewtonDescent, DefectControl, DescentResult,
    DiagonalOperator, DiscreteCallback, DiscreteFunction, DiscreteProblem, Dogleg,
    DynamicalBVPFunction, DynamicalDDEFunction, DynamicalDDEProblem, DynamicalODEFunction,
    DynamicalODEProblem, DynamicalSDEFunction, DynamicalSDEProblem, EigenvalueProblem,
    EigenvalueSolution, EigenvalueTarget, EisenstatWalkerForcing2, EnsembleAnalysis,
    EnsembleContext, EnsembleDistributed, EnsembleProblem, EnsembleSerial, EnsembleSolution,
    EnsembleSplitThreads, EnsembleSummary, EnsembleTestSolution, EnsembleThreads,
    FastShortcutNLLSPolyalg, FunctionOperator, GaussNewton, GeneralizedFirstOrderAlgorithm,
    GeodesicAcceleration, GlobalErrorControl, HOErrorControl, HomotopyNonlinearFunction,
    HomotopyPolyAlgorithm, HomotopyProblem, HomotopySweep, HybridErrorControl,
    IdentityOperator, ImplicitDiscreteFunction, ImplicitDiscreteProblem,
    IncrementingODEFunction, IncrementingODEProblem, IntegralFunction, IntegralProblem,
    IntegralSolution, IntervalNonlinearFunction, IntervalNonlinearProblem,
    InvertibleOperator, KantorovichHomotopy, LevenbergMarquardt, LinearAliasSpecifier,
    LinearProblem, LinearSolution, MatrixOperator, MultiObjectiveOptimizationFunction,
    NewtonDescent, NewtonRaphson, NoAutoDiff, NoAutoDiffSelectedError, NoErrorControl,
    NoiseProblem, NonlinearFunction, NonlinearLeastSquaresProblem, NonlinearProblem,
    NonlinearSolution, NonlinearSolveBase, NonlinearSolveFirstOrder,
    NonlinearSolvePolyAlgorithm, NonlinearVerbosity, NormTerminationMode, NullOperator,
    ODEAliasSpecifier, ODEFunction, ODEInputFunction, ODEProblem, ODESolution,
    OptimizationFunction, OptimizationProblem, OptimizationSolution, PDENoTimeSolution,
    PDEProblem, PDETimeSeriesSolution, PseudoTransient, REErrorControl, RODEFunction,
    RODEProblem, RODESolution, RadiusUpdateSchemes, RelNormSafeBestTerminationMode,
    RelNormSafeTerminationMode, RelNormTerminationMode, RelTerminationMode, ReturnCode,
    RobustMultiNewton, SCCNonlinearProblem, SDDEFunction, SDDEProblem, SDEFunction,
    SDEProblem, SampledIntegralProblem, ScalarOperator, SciMLBase, SciMLOperators,
    SecondOrderBVProblem, SecondOrderDDEProblem, SecondOrderODEProblem,
    SequentialErrorControl, SplitFunction, SplitODEProblem, SplitSDEFunction,
    SplitSDEProblem, StaticWOperator, SteadyStateProblem, SteadyStateSolution,
    SteepestDescent, TensorProductOperator, TensorSumOperator, TimeDomain, TraceAll,
    TraceMinimal, TraceWithJacobianConditionNumber, TrustRegion, TwoPointBVPFunction,
    TwoPointBVProblem, TwoPointDynamicalBVPFunction, TwoPointSecondOrderBVProblem,
    VectorContinuousCallback, WOperator, _process_verbose_param, add_saveat!, add_tstop!,
    addat!, addat_non_user_cache!, addsteps!, auto_dt_reset!, cache_operator,
    change_t_via_interpolation!, check_error, check_keywords, column_coloring, concretize,
    deleteat!, deleteat_non_user_cache!, derivative_discontinuity!, discretize, du_cache,
    first_tstop, full_cache, get_dt, get_du, get_du!, get_proposed_dt, get_rng,
    get_tmp_cache, has_adjoint, has_concretization, has_exp, has_expmv, has_expmv!,
    has_ldiv, has_ldiv!, has_mul, has_mul!, has_rng, has_tstop, hessian_sparsity, init,
    integral, is_discrete_time_domain, iscached, isclock, isconstant, iscontinuous,
    isconvertible, isdiscrete, isinplace, islinear, issolverstepclock, issquare,
    jacobian_sparsity, kronsum, pickchunksize, pop_tstop!, rand_cache, ratenoise_cache,
    reeval_internals_due_to_modification!, reinit!, remake, resize_non_user_cache!,
    row_coloring, savevalues!, set_abstol!, set_proposed_dt!, set_reltol!, set_rng!, set_t!,
    set_u!, solve, solve!, step!, supports_solve_rng, symbolic_discretize,
    symmetric_coloring, terminate!, u_cache, u_modified!, update_coefficients,
    update_coefficients!, user_cache, warn_compat

end # module BoundaryValueDiffEqShooting
