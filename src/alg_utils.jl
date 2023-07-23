alg_order(alg::GeneralMIRK4) = 4
alg_order(alg::GeneralMIRK5) = 5
alg_order(alg::GeneralMIRK6) = 6
alg_order(alg::MIRK4) = 4
alg_order(alg::MIRK5) = 5
alg_order(alg::MIRK6) = 6

alg_stage(alg::GeneralMIRK4) = 3
alg_stage(alg::GeneralMIRK5) = 4
alg_stage(alg::GeneralMIRK6) = 5
alg_stage(alg::MIRK4) = 3
alg_stage(alg::MIRK5) = 4
alg_stage(alg::MIRK6) = 5

SciMLBase.isautodifferentiable(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::BoundaryValueDiffEqAlgorithm) = true
SciMLBase.allowscomplex(alg::BoundaryValueDiffEqAlgorithm) = true

function SciMLBase.isadaptive(alg::Union{
    GeneralMIRK4,
    GeneralMIRK5,
    GeneralMIRK6,
    MIRK4,
    MIRK5,
    MIRK6,
})
    true
end
