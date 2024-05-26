Base.@deprecate MIRKJacobianComputationAlgorithm(
    diffmode = missing; collocation_diffmode = missing, bc_diffmode = missing) BVPJacobianAlgorithm(
    diffmode; nonbc_diffmode = collocation_diffmode, bc_diffmode)
