for stage in (1, 2, 3, 4, 5, 6, 7)
    alg = Symbol("Ascher$(stage)")
    @eval alg_stage(::$(alg)) = $stage
end

for stage in (1, 2, 3, 4, 5, 6, 7)
    alg = Symbol("Ascher$(stage)")
    @eval alg_order(::$(alg)) = $(2 * stage)
end

SciMLBase.isadaptive(alg::AbstractAscher) = true
