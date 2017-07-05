abstract type AbstractMIRKCache end

immutable MIRK4Cache{kType,jType} <: AbstractMIRKCache
    K::kType
    J::jType
end

function alg_cache{T,U}(alg::MIRK4, S::BVPSystem{T,U})
    # k1, k2, k3, k4 = (U(S.M) for i in 1:4)
    # d1, d2, d3, d4 = (U(S.M) for i in 1:4)
    # MIRK4Cache(k1,k2,k3,k4,d1,d2,d3,d4)
    MIRK4Cache([U(S.M) for i in 1:4], [similar(S.y, S.M, S.M) for i in 1:4])
end
