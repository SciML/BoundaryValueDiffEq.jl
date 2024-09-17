# Algorithms
abstract type AbstractMIRK <: BoundaryValueDiffEqAlgorithm end

for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(), jac_alg = BVPJacobianAlgorithm(),
                    defect_threshold = 0.1, max_num_subintervals = 3000)

        $($order)th order Monotonic Implicit Runge Kutta method.

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
        @article{Enright1996RungeKuttaSW,
            title={Runge-Kutta Software with Defect Control for Boundary Value ODEs},
            author={Wayne H. Enright and Paul H. Muir},
            journal={SIAM J. Sci. Comput.},
            year={1996},
            volume={17},
            pages={479-497}
        }
        ```
        """
        @kwdef struct $(alg){N, J <: BVPJacobianAlgorithm, T} <: AbstractMIRK
            nlsolve::N = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end
