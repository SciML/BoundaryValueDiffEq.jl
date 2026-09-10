# Algorithms
abstract type AbstractFIRK <: AbstractBoundaryValueDiffEqAlgorithm end

for stage in (1, 2, 3, 5, 7)
    alg = Symbol("RadauIIa$(stage)")

    @eval begin
        """
            $($alg)(; nlsolve = nothing, optimize = nothing,
                jac_alg = BVPJacobianAlgorithm(), nested_nlsolve = false,
                nested_nlsolve_kwargs = (;), defect_threshold = 0.1,
                max_num_subintervals = 3000) -> $($alg)

        Configures the $($stage)-stage Radau IIA fully implicit Runge-Kutta method.

        ## Keywords

          - `nlsolve = nothing`: nonlinear solver for the collocation residual. The BVP
            Jacobian configuration takes precedence over an autodiff setting on this solver.
          - `optimize = nothing`: optimization solver used when the selected BVP path
            formulates the residual as an optimization problem.
          - `jac_alg = BVPJacobianAlgorithm()`: differentiation strategy for the boundary
            and collocation residuals.

              + For `TwoPointBVProblem`, only `diffmode` is used (defaults to
                `AutoSparse(AutoForwardDiff())` if possible, otherwise
                `AutoSparse(AutoFiniteDiff())`).
              + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
                `nonbc_diffmode`, the default is `AutoSparse(AutoForwardDiff())` if possible,
                otherwise `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, the default is
                `AutoForwardDiff()` if possible, otherwise `AutoFiniteDiff()`.
          - `nested_nlsolve = false`: solve each implicit Runge-Kutta step with a nested
            nonlinear solve instead of including its stages in the global residual.
          - `nested_nlsolve_kwargs = (;)`: keyword arguments forwarded to the nested
            nonlinear solver.
          - `defect_threshold = 0.1`: defect threshold used by mesh adaptivity.
          - `max_num_subintervals = 3000`: maximum number of mesh subintervals.

        ## Fields

          - `nlsolve`: configured nonlinear solver or `nothing`.
          - `optimize`: configured optimization solver or `nothing`.
          - `jac_alg::BVPJacobianAlgorithm`: Jacobian configuration.
          - `nested_nlsolve::Bool`: whether nested nonlinear solves are enabled.
          - `nested_nlsolve_kwargs::NamedTuple`: options for the nested nonlinear solver.
          - `defect_threshold`: adaptive defect threshold.
          - `max_num_subintervals::Int`: mesh-size limit.

        ## Returns

          - `$($alg)`: an algorithm object accepted by `SciMLBase.solve` for a boundary
            value problem.

        ## Examples

        ```jldoctest
        using BoundaryValueDiffEqFIRK: $($alg)

        alg = $($alg)()
        @assert alg isa $($alg)
        # output
        ```

        !!! note

            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

        ## References

        Reference for Lobatto and Radau methods:

        ```bibtex
        @incollection{Jay2015,
            author="Jay, Laurent O.",
            editor="Engquist, Bj{\"o}rn",
            title="Lobatto Methods",
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            year="2015",
            publisher="Springer Berlin Heidelberg",
        }
        @incollection{engquist_radau_2015,
            author = {Hairer, Ernst and Wanner, Gerhard},
            editor={Engquist, Bj{\"o}rn},
            title = {Radau {Methods}},
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            publisher = {Springer Berlin Heidelberg},
            year = {2015},
        }
        ```

        References for implementation of defect control, based on the `bvp5c` solver in MATLAB:

        ```bibtex
        @article{shampine_solving_nodate,
            title = {Solving {Boundary} {Value} {Problems} for {Ordinary} {Differential} {Equations} in {Matlab} with bvp4c},
            author = {Shampine, Lawrence F and Kierzenka, Jacek and Reichelt, Mark W},
            year = {2000},
        }

        @article{kierzenka_bvp_2008,
            title = {A {BVP} {Solver} that {Controls} {Residual} and {Error}},
            author = {Kierzenka, J and Shampine, L F},
            year = {2008},
        }

        @article{russell_adaptive_1978,
            title = {Adaptive {Mesh} {Selection} {Strategies} for {Solving} {Boundary} {Value} {Problems}},
            journal = {SIAM Journal on Numerical Analysis},
            author = {Russell, R. D. and Christiansen, J.},
            year = {1978},
        }
        ```
        """
        Base.@kwdef struct $(alg){N, O, J <: BVPJacobianAlgorithm, T} <: AbstractFIRK
            nlsolve::N = nothing
            optimize::O = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            nested_nlsolve::Bool = false
            nested_nlsolve_kwargs::NamedTuple = (;)
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end

for stage in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIa$(stage)")

    @eval begin
        """
            $($alg)(; nlsolve = nothing, optimize = nothing,
                jac_alg = BVPJacobianAlgorithm(), nested_nlsolve = false,
                nested_nlsolve_kwargs = (;), defect_threshold = 0.1,
                max_num_subintervals = 3000) -> $($alg)

        Configures the $($stage)-stage Lobatto IIIA fully implicit Runge-Kutta method.

        ## Keywords

          - `nlsolve = nothing`: nonlinear solver for the collocation residual. The BVP
            Jacobian configuration takes precedence over an autodiff setting on this solver.
          - `optimize = nothing`: optimization solver used when the selected BVP path
            formulates the residual as an optimization problem.
          - `jac_alg = BVPJacobianAlgorithm()`: differentiation strategy for the boundary
            and collocation residuals.

              + For `TwoPointBVProblem`, only `diffmode` is used (defaults to
                `AutoSparse(AutoForwardDiff())` if possible, otherwise
                `AutoSparse(AutoFiniteDiff())`).
              + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
                `nonbc_diffmode`, the default is `AutoSparse(AutoForwardDiff())` if possible,
                otherwise `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, the default is
                `AutoForwardDiff()` if possible, otherwise `AutoFiniteDiff()`.
          - `nested_nlsolve = false`: solve each implicit Runge-Kutta step with a nested
            nonlinear solve instead of including its stages in the global residual.
          - `nested_nlsolve_kwargs = (;)`: keyword arguments forwarded to the nested
            nonlinear solver.
          - `defect_threshold = 0.1`: defect threshold used by mesh adaptivity.
          - `max_num_subintervals = 3000`: maximum number of mesh subintervals.

        ## Fields

          - `nlsolve`: configured nonlinear solver or `nothing`.
          - `optimize`: configured optimization solver or `nothing`.
          - `jac_alg::BVPJacobianAlgorithm`: Jacobian configuration.
          - `nested_nlsolve::Bool`: whether nested nonlinear solves are enabled.
          - `nested_nlsolve_kwargs::NamedTuple`: options for the nested nonlinear solver.
          - `defect_threshold`: adaptive defect threshold.
          - `max_num_subintervals::Int`: mesh-size limit.

        ## Returns

          - `$($alg)`: an algorithm object accepted by `SciMLBase.solve` for a boundary
            value problem.

        ## Examples

        ```jldoctest
        using BoundaryValueDiffEqFIRK: $($alg)

        alg = $($alg)()
        @assert alg isa $($alg)
        # output
        ```

        !!! note

            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

        ## References

        Reference for Lobatto and Radau methods:

        ```bibtex
        @Inbook{Jay2015,
            author="Jay, Laurent O.",
            editor="Engquist, Bj{\"o}rn",
            title="Lobatto Methods",
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            year="2015",
            publisher="Springer Berlin Heidelberg",
        }
        @incollection{engquist_radau_2015,
            author = {Hairer, Ernst and Wanner, Gerhard},
            title = {Radau {Methods}},
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            publisher = {Springer Berlin Heidelberg},
            editor="Engquist, Bj{\"o}rn",
            year = {2015},
        }
        ```

        References for implementation of defect control, based on the `bvp5c` solver in MATLAB:

        ```bibtex
        @article{shampine_solving_nodate,
            title = {Solving {Boundary} {Value} {Problems} for {Ordinary} {Differential} {Equations} in {Matlab} with bvp4c},
            author = {Shampine, Lawrence F and Kierzenka, Jacek and Reichelt, Mark W},
            year = {2000},
        }

        @article{kierzenka_bvp_2008,
            title = {A {BVP} {Solver} that {Controls} {Residual} and {Error}},
            author = {Kierzenka, J and Shampine, L F},
            year = {2008},
        }

        @article{russell_adaptive_1978,
            title = {Adaptive {Mesh} {Selection} {Strategies} for {Solving} {Boundary} {Value} {Problems}},
            journal = {SIAM Journal on Numerical Analysis},
            author = {Russell, R. D. and Christiansen, J.},
            year = {1978},
        }
        ```
        """
        Base.@kwdef struct $(alg){N, O, J <: BVPJacobianAlgorithm, T} <: AbstractFIRK
            nlsolve::N = nothing
            optimize::O = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            nested_nlsolve::Bool = false
            nested_nlsolve_kwargs::NamedTuple = (;)
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end

for stage in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIb$(stage)")

    @eval begin
        """
            $($alg)(; nlsolve = nothing, optimize = nothing,
                jac_alg = BVPJacobianAlgorithm(), nested_nlsolve = false,
                nested_nlsolve_kwargs = (;), defect_threshold = 0.1,
                max_num_subintervals = 3000) -> $($alg)

        Configures the $($stage)-stage Lobatto IIIB fully implicit Runge-Kutta method.

        ## Keywords

          - `nlsolve = nothing`: nonlinear solver for the collocation residual. The BVP
            Jacobian configuration takes precedence over an autodiff setting on this solver.
          - `optimize = nothing`: optimization solver used when the selected BVP path
            formulates the residual as an optimization problem.
          - `jac_alg = BVPJacobianAlgorithm()`: differentiation strategy for the boundary
            and collocation residuals.

              + For `TwoPointBVProblem`, only `diffmode` is used (defaults to
                `AutoSparse(AutoForwardDiff())` if possible, otherwise
                `AutoSparse(AutoFiniteDiff())`).
              + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
                `nonbc_diffmode`, the default is `AutoSparse(AutoForwardDiff())` if possible,
                otherwise `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, the default is
                `AutoForwardDiff()` if possible, otherwise `AutoFiniteDiff()`.
          - `nested_nlsolve = false`: solve each implicit Runge-Kutta step with a nested
            nonlinear solve instead of including its stages in the global residual.
          - `nested_nlsolve_kwargs = (;)`: keyword arguments forwarded to the nested
            nonlinear solver.
          - `defect_threshold = 0.1`: defect threshold used by mesh adaptivity.
          - `max_num_subintervals = 3000`: maximum number of mesh subintervals.

        ## Fields

          - `nlsolve`: configured nonlinear solver or `nothing`.
          - `optimize`: configured optimization solver or `nothing`.
          - `jac_alg::BVPJacobianAlgorithm`: Jacobian configuration.
          - `nested_nlsolve::Bool`: whether nested nonlinear solves are enabled.
          - `nested_nlsolve_kwargs::NamedTuple`: options for the nested nonlinear solver.
          - `defect_threshold`: adaptive defect threshold.
          - `max_num_subintervals::Int`: mesh-size limit.

        ## Returns

          - `$($alg)`: an algorithm object accepted by `SciMLBase.solve` for a boundary
            value problem.

        ## Examples

        ```jldoctest
        using BoundaryValueDiffEqFIRK: $($alg)

        alg = $($alg)()
        @assert alg isa $($alg)
        # output
        ```

        !!! note

            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

        ## References

        Reference for Lobatto and Radau methods:

        ```bibtex
        @Inbook{Jay2015,
            author="Jay, Laurent O.",
            editor="Engquist, Bj{\"o}rn",
            title="Lobatto Methods",
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            year="2015",
            publisher="Springer Berlin Heidelberg",
        }
        @incollection{engquist_radau_2015,
            author = {Hairer, Ernst and Wanner, Gerhard},
            title = {Radau {Methods}},
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            publisher = {Springer Berlin Heidelberg},
            editor="Engquist, Bj{\"o}rn",
            year = {2015},
        }
        ```

        References for implementation of defect control, based on the `bvp5c` solver in MATLAB:

        ```bibtex
        @article{shampine_solving_nodate,
            title = {Solving {Boundary} {Value} {Problems} for {Ordinary} {Differential} {Equations} in {Matlab} with bvp4c},
            author = {Shampine, Lawrence F and Kierzenka, Jacek and Reichelt, Mark W},
            year = {2000},
        }

        @article{kierzenka_bvp_2008,
            title = {A {BVP} {Solver} that {Controls} {Residual} and {Error}},
            author = {Kierzenka, J and Shampine, L F},
            year = {2008},
        }

        @article{russell_adaptive_1978,
            title = {Adaptive {Mesh} {Selection} {Strategies} for {Solving} {Boundary} {Value} {Problems}},
            journal = {SIAM Journal on Numerical Analysis},
            author = {Russell, R. D. and Christiansen, J.},
            year = {1978},
        }
        ```
        """
        Base.@kwdef struct $(alg){N, O, J <: BVPJacobianAlgorithm, T} <: AbstractFIRK
            nlsolve::N = nothing
            optimize::O = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            nested_nlsolve::Bool = false
            nested_nlsolve_kwargs::NamedTuple = (;)
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end

for stage in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIc$(stage)")

    @eval begin
        """
            $($alg)(; nlsolve = nothing, optimize = nothing,
                jac_alg = BVPJacobianAlgorithm(), nested_nlsolve = false,
                nested_nlsolve_kwargs = (;), defect_threshold = 0.1,
                max_num_subintervals = 3000) -> $($alg)

        Configures the $($stage)-stage Lobatto IIIC fully implicit Runge-Kutta method.

        ## Keywords

          - `nlsolve = nothing`: nonlinear solver for the collocation residual. The BVP
            Jacobian configuration takes precedence over an autodiff setting on this solver.
          - `optimize = nothing`: optimization solver used when the selected BVP path
            formulates the residual as an optimization problem.
          - `jac_alg = BVPJacobianAlgorithm()`: differentiation strategy for the boundary
            and collocation residuals.

              + For `TwoPointBVProblem`, only `diffmode` is used (defaults to
                `AutoSparse(AutoForwardDiff())` if possible, otherwise
                `AutoSparse(AutoFiniteDiff())`).
              + For `BVProblem`, `bc_diffmode` and `nonbc_diffmode` are used. For
                `nonbc_diffmode`, the default is `AutoSparse(AutoForwardDiff())` if possible,
                otherwise `AutoSparse(AutoFiniteDiff())`. For `bc_diffmode`, the default is
                `AutoForwardDiff()` if possible, otherwise `AutoFiniteDiff()`.
          - `nested_nlsolve = false`: solve each implicit Runge-Kutta step with a nested
            nonlinear solve instead of including its stages in the global residual.
          - `nested_nlsolve_kwargs = (;)`: keyword arguments forwarded to the nested
            nonlinear solver.
          - `defect_threshold = 0.1`: defect threshold used by mesh adaptivity.
          - `max_num_subintervals = 3000`: maximum number of mesh subintervals.

        ## Fields

          - `nlsolve`: configured nonlinear solver or `nothing`.
          - `optimize`: configured optimization solver or `nothing`.
          - `jac_alg::BVPJacobianAlgorithm`: Jacobian configuration.
          - `nested_nlsolve::Bool`: whether nested nonlinear solves are enabled.
          - `nested_nlsolve_kwargs::NamedTuple`: options for the nested nonlinear solver.
          - `defect_threshold`: adaptive defect threshold.
          - `max_num_subintervals::Int`: mesh-size limit.

        ## Returns

          - `$($alg)`: an algorithm object accepted by `SciMLBase.solve` for a boundary
            value problem.

        ## Examples

        ```jldoctest
        using BoundaryValueDiffEqFIRK: $($alg)

        alg = $($alg)()
        @assert alg isa $($alg)
        # output
        ```

        !!! note

            For type-stability, the chunksizes for ForwardDiff ADTypes in
            `BVPJacobianAlgorithm` must be provided.

        ## References

        Reference for Lobatto and Radau methods:

        ```bibtex
        @Inbook{Jay2015,
            author="Jay, Laurent O.",
            editor="Engquist, Bj{\"o}rn",
            title="Lobatto Methods",
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            year="2015",
            publisher="Springer Berlin Heidelberg",
        }
        @incollection{engquist_radau_2015,
            author = {Hairer, Ernst and Wanner, Gerhard},
            title = {Radau {Methods}},
            booktitle = {Encyclopedia of {Applied} and {Computational} {Mathematics}},
            publisher = {Springer Berlin Heidelberg},
            editor="Engquist, Bj{\"o}rn",
            year = {2015},
        }
        ```

        References for implementation of defect control, based on the `bvp5c` solver in MATLAB:

        ```bibtex
        @article{shampine_solving_nodate,
            title = {Solving {Boundary} {Value} {Problems} for {Ordinary} {Differential} {Equations} in {Matlab} with bvp4c},
            author = {Shampine, Lawrence F and Kierzenka, Jacek and Reichelt, Mark W},
            year = {2000},
        }

        @article{kierzenka_bvp_2008,
            title = {A {BVP} {Solver} that {Controls} {Residual} and {Error}},
            author = {Kierzenka, J and Shampine, L F},
            year = {2008},
        }

        @article{russell_adaptive_1978,
            title = {Adaptive {Mesh} {Selection} {Strategies} for {Solving} {Boundary} {Value} {Problems}},
            journal = {SIAM Journal on Numerical Analysis},
            author = {Russell, R. D. and Christiansen, J.},
            year = {1978},
        }
        ```
        """
        Base.@kwdef struct $(alg){N, O, J <: BVPJacobianAlgorithm, T} <: AbstractFIRK
            nlsolve::N = nothing
            optimize::O = nothing
            jac_alg::J = BVPJacobianAlgorithm()
            nested_nlsolve::Bool = false
            nested_nlsolve_kwargs::NamedTuple = (;)
            defect_threshold::T = 0.1
            max_num_subintervals::Int = 3000
        end
    end
end

# FIRK Algorithms that don't use adaptivity
const FIRKNoAdaptivity = Union{LobattoIIIb2, RadauIIa1, LobattoIIIc2}
