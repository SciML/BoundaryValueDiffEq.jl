alg_order(alg::GeneralMIRK4) = 4
alg_order(alg::GeneralMIRK6) = 6
alg_order(alg::MIRK4) = 4
alg_order(alg::MIRK6) = 6

SciMLBase.isautodifferentiable(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allowscomplex(alg::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.isadaptive(alg::Union{GeneralMIRK4, GeneralMIRK6, MIRK4, MIRK6}) = false
