abstract type AbstractMIRKCache end
abstract type MIRKCache <: AbstractMIRKCache end

const AA3 = AbstractArray{T, 3} where {T}

for order in (3, 4, 5, 6)
    cache = Symbol("MIRK$(order)Cache")
    # `k_discrete` stores discrete stages for each subinterval,
    # hence the size of k_discrete is M × stage × (N - 1)
    @eval struct $(cache){kType <: AA3} <: MIRKCache
        k_discrete::kType
    end

    @eval @truncate_stacktrace $cache

    algType = Symbol("MIRK$(order)")
    @eval function alg_cache(::$algType, S::BVPSystem)
        return $(cache)(similar(S.tmp, S.M, S.stage, S.N - 1))
    end
end
