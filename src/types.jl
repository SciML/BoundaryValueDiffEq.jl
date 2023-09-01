# MIRK Method Tableaus
struct MIRKTableau{sType, cType, vType, bType, xType}
    """Discrete stages of MIRK formula"""
    s::sType
    c::cType
    v::vType
    b::bType
    x::xType

    function MIRKTableau(s, c, v, b, x)
        @assert eltype(c) == eltype(v) == eltype(b) == eltype(x)
        return new{typeof(s), typeof(c), typeof(v), typeof(b), typeof(x)}(s, c, v, b, x)
    end
end

@truncate_stacktrace MIRKTableau 1

struct MIRKInterpTableau{s, c, v, x, τ}
    s_star::s
    c_star::c
    v_star::v
    x_star::x
    τ_star::τ

    function MIRKInterpTableau(s_star, c_star, v_star, x_star, τ_star)
        @assert eltype(c_star) == eltype(v_star) == eltype(x_star)
        return new{typeof(s_star), typeof(c_star), typeof(v_star), typeof(x_star),
            typeof(τ_star)}(s_star,
            c_star, v_star, x_star, τ_star)
    end
end

@truncate_stacktrace MIRKInterpTableau 1

# ODE BVP problem system
## NOTE: We might want to decouple this type from MIRK sometime later
# FIXME: Remove
struct BVPSystem{F <: Function, B <: Union{Function, SciMLBase.TwoPointBVPFunction},
    tType <: DiffCache}
    order::Int                  # The order of MIRK method
    stage::Int                  # The state of MIRK method
    M::Int                      # Number of equations in the ODE system
    N::Int                      # Number of nodes in the mesh
    f!::F                       # M -> M
    bc!::B                      # 2 -> 2
    tmp::tType                  # M
end

Base.eltype(S::BVPSystem) = eltype(S.tmp.du)

# Sparsity Detection
@static if VERSION < v"1.9"
    # Sparse Linear Solvers in LinearSolve.jl are a bit flaky on older versions
    Base.@kwdef struct MIRKJacobianComputationAlgorithm{BD, CD}
        bc_diffmode::BD = AutoForwardDiff()
        collocation_diffmode::CD = AutoForwardDiff()
    end
else
    Base.@kwdef struct MIRKJacobianComputationAlgorithm{BD, CD}
        bc_diffmode::BD = AutoForwardDiff()
        collocation_diffmode::CD = AutoSparseForwardDiff()
    end
end
