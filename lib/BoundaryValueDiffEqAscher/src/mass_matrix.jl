# Ascher's collocation assumes the semi-explicit form `M = [I 0; 0 0]`, with the
# differential variables first and the algebraic ones last. A constant mass matrix in
# which every row and every column has at most one nonzero is that form up to a row
# permutation, a column permutation and a row scaling. ModelingToolkit produces such
# matrices (`M[eq, var] = 1` for each differential equation), but it may list algebraic
# variables first or interleave them with the differential ones.

"""
    __ascher_semi_explicit_permutation(mass_matrix)

Return `nothing` when `mass_matrix` is already in the canonical form `[I 0; 0 0]`, or a
named tuple `(; pv, pe, scale, ndiff)` describing how to reorder it. Internal variable
`i` is user variable `pv[i]`, internal equation `i` is user equation `pe[i]` scaled by
`scale[i]`, and the first `ndiff` internal variables and equations are differential.
Throws an `ArgumentError` for mass matrices that are not semi-explicit up to such a
reordering.
"""
__ascher_semi_explicit_permutation(::LinearAlgebra.UniformScaling) = nothing

function __ascher_semi_explicit_permutation(mass_matrix::AbstractMatrix)
    n = LinearAlgebra.checksquare(mass_matrix)
    diff_eqs = Int[]
    diff_vars = Int[]
    for i in 1:n
        nonzeros = findall(!iszero, view(mass_matrix, i, :))
        isempty(nonzeros) && continue
        length(nonzeros) == 1 || throw(
            ArgumentError(
                "Ascher solvers support semi-explicit mass matrices only: row $i of the " *
                    "mass matrix has $(length(nonzeros)) nonzero entries, at most one is allowed."
            )
        )
        push!(diff_eqs, i)
        push!(diff_vars, only(nonzeros))
    end
    allunique(diff_vars) || throw(
        ArgumentError(
            "Ascher solvers support semi-explicit mass matrices only: some column of " *
                "the mass matrix has more than one nonzero entry."
        )
    )
    ndiff = length(diff_eqs)
    pe = vcat(diff_eqs, setdiff(1:n, diff_eqs))
    pv = vcat(diff_vars, setdiff(1:n, diff_vars))
    S = float(eltype(mass_matrix))
    scale = [i <= ndiff ? inv(S(mass_matrix[pe[i], pv[i]])) : one(S) for i in 1:n]
    if pe == 1:n && pv == 1:n && all(isone, scale)
        return nothing
    end
    return (; pv, pe, scale, ndiff)
end

# The user's right-hand side seen in the internal ordering.
struct __AscherPermutedRHS{F, S}
    f::F
    pv::Vector{Int}
    pe::Vector{Int}
    scale::S
end

function (rhs::__AscherPermutedRHS)(du, u, p, t)
    u_user = similar(u)
    u_user[rhs.pv] .= vec(u)
    du_user = similar(du)
    rhs.f(du_user, u_user, p, t)
    for i in eachindex(rhs.pe)
        du[i] = du_user[rhs.pe[i]] * rhs.scale[i]
    end
    return nothing
end

function (rhs::__AscherPermutedRHS)(u, p, t)
    u_user = similar(u)
    u_user[rhs.pv] .= vec(u)
    du_user = rhs.f(u_user, p, t)
    return [du_user[rhs.pe[i]] * rhs.scale[i] for i in eachindex(rhs.pe)]
end

__bvp_nlls(::BVProblem{U, T, iip, nlls}) where {U, T, iip, nlls} = nlls

"""
    __ascher_permuted_problem(prob, perm)

Rebuild `prob` in the internal ordering described by `perm`, with the canonical mass
matrix `Diagonal([ones(ndiff); zeros(n - ndiff)])`. Boundary conditions and the cost keep
the user's ordering: Ascher evaluates them on an `AscherBoundarySolution`, which maps
back to the user's ordering when called.
"""
function __ascher_permuted_problem(prob::BVProblem, perm)
    (; pv, pe, scale, ndiff) = perm
    prob.u0 isa AbstractVector{<:Number} || throw(
        ArgumentError(
            "Ascher solvers need a vector `u0` when the mass matrix has to be reordered " *
                "into semi-explicit form; got $(typeof(prob.u0))."
        )
    )
    n = length(pv)
    length(prob.u0) == n || throw(
        ArgumentError(
            "Ascher solvers need the mass matrix to cover the whole state when it has to be " *
                "reordered into semi-explicit form: `u0` has $(length(prob.u0)) entries but the " *
                "mass matrix is $n × $n (e.g. control inputs appended to the state)."
        )
    )
    iip = isinplace(prob)
    mass_matrix = LinearAlgebra.Diagonal([ones(ndiff); zeros(n - ndiff)])
    twopoint = prob.problem_type isa TwoPointBVProblem
    # A two-point `BVPFunction` wants the pair of endpoint prototypes, while the stored
    # one is an `ArrayPartition`. When it is missing, infer it here from the user's
    # problem: inferring it later would call the boundary functions on the reordered `u0`.
    bcresid_prototype = twopoint ?
        first(__get_bcresid_prototype(prob.problem_type, prob, prob.u0)) :
        prob.f.bcresid_prototype
    fun = BVPFunction{iip}(
        __AscherPermutedRHS(prob.f, pv, pe, scale), prob.f.bc;
        mass_matrix, twopoint = Val(twopoint), cost = prob.f.cost, bcresid_prototype
    )
    nlls = __bvp_nlls(prob)
    return BVProblem{iip}(
        fun, prob.u0[pv], prob.tspan, prob.p;
        prob.lb, prob.ub, prob.lcons, prob.ucons, prob.problem_type,
        prob.singular_term, nlls = nlls isa Bool ? Val(nlls) : nothing, prob.kwargs...
    )
end

# Map a vector from the internal ordering back to the user's ordering.
__ascher_unpermute(u, ::Nothing) = u
function __ascher_unpermute(u, pv::Vector{Int})
    u_user = similar(u)
    u_user[pv] .= u
    return u_user
end
