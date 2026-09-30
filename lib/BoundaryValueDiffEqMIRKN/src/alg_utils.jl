for order in (4, 6)
    alg = Symbol("MIRKN$(order)")
    @eval alg_order(::$(alg)) = $order
    @eval alg_stage(::$(alg)) = $(order - 1)
end

SciMLBase.isadaptive(alg::AbstractMIRKN) = false
