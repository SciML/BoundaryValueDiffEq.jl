alg_order(alg::MIRK4) = 4
alg_order(alg::GeneralMIRK4) = 4

SciMLBase.isautodifferentiable(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allowscomplex(alg::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.isadaptive(alg::Union{MIRK4,GeneralMIRK4}) = false
