for order in (2, 3, 4, 5, 6)
    alg = Symbol("MIRK$(order)")
    @eval alg_order(::$(alg)) = $order
    @eval alg_stage(::$(alg)) = $(order - 1)
end

SciMLBase.isautodifferentiable(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allowscomplex(alg::BoundaryValueDiffEqAlgorithm) = true

SciMLBase.isadaptive(alg::AbstractMIRK) = true
