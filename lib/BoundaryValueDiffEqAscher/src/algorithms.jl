abstract type AbstractAscher <: BoundaryValueDiffEqAlgorithm end

for stage in (1, 2, 3, 4, 5, 6, 7)
    alg = Symbol("Ascher$(stage)")

    @eval begin
        """
            $($alg)(; nlsolve = NewtonRaphson(), max_num_subintervals = 3000)

        $($stage)th stage Gauss Legendre collocation methods with adaptivity adapted from Ascher's implementation.

        ## Keyword Arguments

          - `nlsolve`: Internal Nonlinear solver. Any solver which conforms to the SciML
            `NonlinearProblem` interface can be used. Note that any autodiff argument for
            the solver will be ignored and a custom jacobian algorithm will be used.
          - `max_num_subintervals`: Number of maximal subintervals, default as 3000.
          - `zeta`: side condition points, should always be provided.

        !!! note
            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

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
        @kwdef struct $(alg){N, J <: BVPJacobianAlgorithm} <: AbstractAscher
            nlsolve::N = nothing
            zeta::Vector{Float64} = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            max_num_subintervals::Int = 3000
        end
    end
end

function concretize_jacobian_algorithm(alg::AbstractAscher, prob)
    @set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
    return alg
end
