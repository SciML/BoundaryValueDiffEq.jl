SciMLBase.isautodifferentiable(::AbstractBoundaryValueDiffEqAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::AbstractBoundaryValueDiffEqAlgorithm) = true
SciMLBase.allowscomplex(alg::AbstractBoundaryValueDiffEqAlgorithm) = true

# Opt-in for SecondOrderBVProblem. Default false; MIRKN overrides to true.
# Third-party algorithms that handle SecondOrderBVProblem via
# `__init(::AbstractBVProblem, ::MyAlg)` must also set this to true.
__supports_second_order(::AbstractBoundaryValueDiffEqAlgorithm) = false
