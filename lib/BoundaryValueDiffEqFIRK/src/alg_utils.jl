for stage in (1, 2, 3, 5, 7)
    alg = Symbol("RadauIIa$(stage)")
    @eval alg_order(::$(alg)) = $(2 * stage - 1)
    @eval alg_stage(::$(alg)) = $stage
end

for stage in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIa$(stage)")
    @eval alg_order(::$(alg)) = $(2 * stage - 2)
    @eval alg_stage(::$(alg)) = $stage
end

for stage in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIb$(stage)")
    @eval alg_order(::$(alg)) = $(2 * stage - 2)
    @eval alg_stage(::$(alg)) = $stage
end

for stage in (2, 3, 4, 5)
    alg = Symbol("LobattoIIIc$(stage)")
    @eval alg_order(::$(alg)) = $(2 * stage - 2)
    @eval alg_stage(::$(alg)) = $stage
end

SciMLBase.isadaptive(alg::AbstractFIRK) = true
