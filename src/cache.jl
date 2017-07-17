abstract type AbstractMIRKCache end

immutable MIRK4Cache{kType,jType} <: AbstractMIRKCache
    K::kType
    LJ::Vector{jType}    # Left strip of the Jacobian
    RJ::Vector{jType}    # Right strip of the Jacobian
    Jacobian::jType
end

function alg_cache{T,U}(alg::MIRK4, S::BVPSystem{T,U})
    # k1, k2, k3, k4 = (U(S.M) for i in 1:4)
    # d1, d2, d3, d4 = (U(S.M) for i in 1:4)
    # MIRK4Cache(k1,k2,k3,k4,d1,d2,d3,d4)
    LJ, RJ = [[similar(S.y[1], S.M, S.M) for i in 1:4] for j in 1:2]
    Jacobian = Matrix{T}(S.M*(S.N+1), S.M*(S.N+1))
    MIRK4Cache([U(S.M) for i in 1:4], LJ, RJ, Jacobian)
end
