abstract type AbstractMIRKCache end
abstract type MIRKCache <: AbstractMIRKCache end
abstract type GeneralMIRKCache <: AbstractMIRKCache end

#=
struct MIRK4Cache{kType,jsType,jType} <: MIRKCache
    K::kType
    LJ::Vector{jsType}    # Left strip of the Jacobian
    RJ::Vector{jsType}    # Right strip of the Jacobian
    Jacobian::jType
end

function alg_cache{T,U}(alg::MIRK4, S::BVPSystem{T,U})
    LJ, RJ = [[similar(S.y[1], S.M, S.M) for i in 1:4] for j in 1:2]
    Jacobian = zeros(T, S.M*S.N, S.M*S.N)
    MIRK4Cache([U(undef, S.M) for i in 1:4], LJ, RJ, Jacobian)
end
=#

struct MIRK4GeneralCache{kType} <: GeneralMIRKCache
    K::kType
end

alg_cache(alg::Union{GeneralMIRK4,MIRK4}, S::BVPSystem{T,U}) where {T,U} = MIRK4GeneralCache([U(undef,S.M) for i in 1:4])
