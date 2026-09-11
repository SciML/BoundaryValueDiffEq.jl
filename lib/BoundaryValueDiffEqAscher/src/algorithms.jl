abstract type AbstractAscher <: AbstractBoundaryValueDiffEqAlgorithm end

for stage in (1, 2, 3, 4, 5, 6, 7)
    alg = Symbol("Ascher$(stage)")

    @eval begin
        """
            $($alg)(; nlsolve = nothing, optimize = nothing, zeta = Float64[],
                jac_alg = BVPJacobianAlgorithm(), max_num_subintervals = 3000)

        $($stage)-stage Gauss-Legendre collocation method with Ascher error-control
        adaptivity and mesh refinement for boundary-value problems, including problems
        with algebraic constraints.

        ## Fields

          - `nlsolve`: Nonlinear solver used for the collocation system. `nothing`
            selects the package default.
          - `optimize`: Optimization solver used by the mesh-refinement machinery.
            `nothing` selects the package default.
          - `zeta`: Side-condition locations for problems that require them. The default
            empty vector is appropriate when no side conditions are present.
          - `jac_alg`: `BVPJacobianAlgorithm` that selects the Jacobian construction
            strategy for the collocation system.
          - `max_num_subintervals`: Maximum number of mesh subintervals permitted while
            refining the solution.

        ## Keyword Arguments

          - `nlsolve = nothing`: Internal nonlinear solver. Any solver implementing the
            SciML `NonlinearProblem` interface may be used. Its autodifferentiation
            setting is ignored because this solver uses `jac_alg`.
          - `optimize = nothing`: Internal optimization solver. Any solver implementing
            the SciML `OptimizationProblem` interface may be used. Its
            autodifferentiation setting is ignored because this solver uses `jac_alg`.
          - `zeta = Float64[]`: Side-condition locations. Supply the points required by
            the problem; leave empty when the problem has no side conditions.
          - `jac_alg = BVPJacobianAlgorithm()`: Jacobian construction strategy. For
            type stability, provide ForwardDiff chunk sizes in the AD types selected by
            this value.
          - `max_num_subintervals = 3000`: Maximum number of mesh subintervals.

        ## Example

        ```julia
        alg = $($alg)(zeta = [0.0, 0.5, 1.0])
        ```

        ## References

        ```bibtex
        @article{Ascher1994CollocationSF,
            title={Collocation Software for Boundary Value Differential-Algebraic Equations},
            author={Uri M. Ascher and Raymond J. Spiteri},
            journal={SIAM J. Sci. Comput.},
            year={1994},
            volume={15},
            pages={938-952},
            url={https://api.semanticscholar.org/CorpusID:10597070}
        }

        @article{Ascher1979ACS,
            title={A collocation solver for mixed order systems of boundary value problems},
            author={Uri M. Ascher and J. Christiansen and Robert D. Russell},
            journal={Mathematics of Computation},
            year={1979},
            volume={33},
            pages={659-679},
            url={https://api.semanticscholar.org/CorpusID:121729124}
        }
        ```
        """
        @kwdef struct $(alg){N, O, J <: BVPJacobianAlgorithm} <: AbstractAscher
            nlsolve::N = nothing
            optimize::O = nothing
            zeta::Vector{Float64} = Float64[]
            jac_alg::J = BVPJacobianAlgorithm()
            max_num_subintervals::Int = 3000
        end
    end
end

function BoundaryValueDiffEqCore.concrete_jacobian_algorithm(
        jac_alg::BVPJacobianAlgorithm, prob::BVProblem, alg::AbstractAscher
    )
    return BVPJacobianAlgorithm(__default_nonsparse_ad(prob.u0))
end
