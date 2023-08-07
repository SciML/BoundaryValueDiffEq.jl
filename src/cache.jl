abstract type AbstractMIRKCache end
abstract type MIRKCache <: AbstractMIRKCache end
abstract type GeneralMIRKCache <: AbstractMIRKCache end

const AA3 = AbstractArray{T, 3} where {T}

for order in (3, 4, 5, 6)
    cache = Symbol("MIRK$(order)GeneralCache")
    # `k_discrete` stores discrete stages for each subinterval,
    # hence the size of k_discrete is M × stage × (N - 1)
    @eval struct $(cache){kType <: AA3} <: GeneralMIRKCache
        k_discrete::kType
    end

    @eval @truncate_stacktrace $cache

    for algType in (Symbol("GeneralMIRK$order"), Symbol("MIRK$order"))
        @eval function alg_cache(::$algType, S::BVPSystem)
            return $(cache)(similar(S.tmp, S.M, S.stage, S.N - 1))
        end
    end
end
