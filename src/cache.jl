abstract type AbstractMIRKCache end

struct MIRK4Cache{kType,jType} <: AbstractMIRKCache
    K::kType
    LJ::Vector{jType}    # Left strip of the Jacobian
    RJ::Vector{jType}    # Right strip of the Jacobian
    Jacobian::jType
end

function alg_cache{T,U}(alg::Union{GeneralMIRK4, MIRK4}, S::BVPSystem{T,U})
    LJ, RJ = [[similar(S.y[1], S.M, S.M) for i in 1:4] for j in 1:2]
    Jacobian = zeros(T, S.M*S.N, S.M*S.N)
    MIRK4Cache([U(S.M) for i in 1:4], LJ, RJ, Jacobian)
end
