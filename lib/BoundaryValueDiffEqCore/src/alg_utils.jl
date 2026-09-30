SciMLBase.isautodifferentiable(::AbstractBoundaryValueDiffEqAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::AbstractBoundaryValueDiffEqAlgorithm) = true
SciMLBase.allowscomplex(alg::AbstractBoundaryValueDiffEqAlgorithm) = true
