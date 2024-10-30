# Algorithms
abstract type AbstractMIRKN <: BoundaryValueDiffEqAlgorithm end

for order in (4, 6)
    alg = Symbol("MIRKN$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm(),
                    defect_threshold = 0.1, max_num_subintervals = 3000)

        $($order)th order Monotonic Implicit Runge Kutta Nyström method.

        ## Keyword Arguments

          - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
            `NonlinearProblem` interface can be used. Note that any autodiff argument for
            the solver will be ignored and a custom jacobian algorithm will be used.
          - `jac_alg`: Jacobian Algorithm used for the nonlinear solver. Defaults to
            `BVPJacobianAlgorithm()`, which automatically decides the best algorithm to
            use based on the input types and problem type.
            - For `TwoPointBVProblem`, only `diffmode` is used (defaults to
              `AutoSparse(AutoForwardDiff())` if possible else `AutoSparse(AutoFiniteDiff())`).
            - For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
              `nonbc_diffmode` defaults to `AutoSparse(AutoForwardDiff())` if possible else
              `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, defaults to `AutoForwardDiff` if
              possible else `AutoFiniteDiff`.
          - `defect_threshold`: Threshold for defect control.
          - `max_num_subintervals`: Number of maximal subintervals, default as 3000.

        !!! note
            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

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
        @kwdef struct $(alg){N, J <: BVPJacobianAlgorithm, T} <: AbstractMIRKN
            nlsolve::N = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end
