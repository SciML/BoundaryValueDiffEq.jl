# Algorithms
abstract type AbstractMIRKN <: AbstractBoundaryValueDiffEqAlgorithm end

for order in (4, 6)
    alg = Symbol("MIRKN$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = nothing, optimize = nothing, jac_alg = BVPJacobianAlgorithm(),
                    defect_threshold = 0.1, max_num_subintervals = 3000)

        $($order)th order Monotonic Implicit Runge Kutta Nyström method.

        ## Fields

        - `nlsolve`: Optional nonlinear solver algorithm. `nothing` selects the package default.
        - `optimize`: Optional optimization solver algorithm. `nothing` disables optimization-based
          initialization.
        - `jac_alg`: Jacobian construction configuration used by the nonlinear solver.
        - `defect_threshold`: Defect-control threshold used to refine the mesh.
        - `max_num_subintervals`: Maximum number of mesh subintervals permitted during refinement.

        ## Keyword Arguments

        - `nlsolve = nothing`: Internal nonlinear solver. Any solver that conforms to the SciML
          `NonlinearProblem` interface can be used. Its autodiff setting is ignored because MIRKN
          uses `jac_alg` to construct the Jacobian.
        - `optimize = nothing`: Internal optimization solver. Any solver that conforms to the
          SciML `OptimizationProblem` interface can be used for initialization. Load the solver
          package before constructing the algorithm.
        - `jac_alg = BVPJacobianAlgorithm()`: Jacobian algorithm used for the nonlinear solver.
          It automatically selects an algorithm from the problem and input types.
          - For `TwoPointBVProblem`, only `diffmode` is used (defaults to
            `AutoSparse(AutoForwardDiff())` if possible else `AutoSparse(AutoFiniteDiff())`).
          - For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
            `nonbc_diffmode` defaults to `AutoSparse(AutoForwardDiff())` if possible else
            `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, defaults to `AutoForwardDiff` if
            possible else `AutoFiniteDiff`.
        - `defect_threshold = 0.1`: Threshold for defect control.
        - `max_num_subintervals = 3000`: Maximum number of mesh subintervals.

        !!! note

            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

        ## Examples

        ```jldoctest
        julia> using BoundaryValueDiffEqMIRKN: MIRKN$($order)

        julia> MIRKN$($order)().max_num_subintervals
        3000
        ```

        ## References

        ```bibtex
        @article{Muir2001MonoImplicitRM,
            title={Mono-Implicit Runge-Kutta-Nystr{\"o}m Methods with Application to Boundary Value Ordinary Differential Equations},
            author={Paul H. Muir and Mark F. Adams},
            journal={BIT Numerical Mathematics},
            year={2001},
            volume={41},
            pages={776-799}
        }
        ```
        """
        @kwdef struct $(alg){N, O, J <: BVPJacobianAlgorithm, T} <: AbstractMIRKN
            nlsolve::N = nothing
            optimize::O = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end
