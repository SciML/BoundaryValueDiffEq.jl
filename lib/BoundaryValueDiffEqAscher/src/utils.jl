function build_almost_block_diagonals(nleft::Integer, ncomp::Integer, mesh, ::Type{T}) where {T}
    ncol = 2 * ncomp
    n = length(mesh) - 1
    # build integs (describing block structure of matrix)
    rows = fill(ncomp + nleft, n)
    cols = repeat([ncol], n)
    lasts = repeat([ncomp], n)
    lasts[end] = ncol
    rows[end] = ncol
    g = IntermediateAlmostBlockDiagonal([zeros(T, rows[i], cols[i]) for i in 1:n], lasts)
    return g
end

function __ascher_linear_interp_state(us, ts, t)
    j = searchsortedlast(ts, t)
    j == 0 && return vec(first(us))
    (j == length(ts) || t == ts[j]) && return vec(us[j])
    θ = (t - ts[j]) / (ts[j + 1] - ts[j])
    left, right = vec(us[j]), vec(us[j + 1])
    return left .+ θ .* (right .- left)
end

"""
    __ascher_guess_evaluator(u0, mesh, p)

Return a function that samples the initial state at mesh nodes and collocation points.
Ascher uses the time points of a timestamped guess as its mesh. Untimed discrete
samples are uniformly spaced over the mesh span, and discrete values between
samples are linearly interpolated. A single state supplies a constant guess.
"""
function __ascher_guess_evaluator(u0, mesh, p)
    us = __initial_guess_on_mesh(u0, (first(mesh),), p).u
    length(us) == 1 && return _ -> vec(only(us))
    ts = hasproperty(u0, :t) ? mesh : range(first(mesh), last(mesh); length = length(us))
    return t -> __ascher_linear_interp_state(us, ts, t)
end
__ascher_guess_evaluator(f::Function, mesh, p) = t -> vec(__initial_guess(f, p, t))

# Custom pivot LU factorization and substitution for simple usage and
# customized array types
function __factorize!(a::Matrix{T}, ipvt::Vector) where {T}
    t = 0.0
    n = size(a, 1)
    if n - 1 >= 1
        for k in 1:(n - 1)
            kp1 = k + 1
            # find l = pivot index
            l = argmax(abs.(a[k:end, k])) + k - 1
            ipvt[k] = l
            # zero pivot implies this column already triangularized
            if a[l, k] !== 0
                # interchange if necessary
                if l !== k
                    t = a[l, k]
                    a[l, k] = a[k, k]
                    a[k, k] = t
                end
                # compute  multipliers
                t = -1.0 / a[k, k]
                @views a[(k + 1):end, k] .= a[(k + 1):end, k] .* t
                # row elimination with column indexing
                for j in kp1:n
                    t = a[l, j]
                    if l !== k
                        a[l, j] = a[k, j]
                        a[k, j] = t
                    end
                    @views __muladd!(t, a[(k + 1):end, k], a[(k + 1):end, j])
                end
            end
        end
    end
    return ipvt[n] = n
end

function __substitute!(a::Matrix{T1}, ipvt::Vector{I}, vb::Vector{Vector{T2}}) where {
        T1, T2, I,
    }
    n = size(a, 1)
    b = reduce(vcat, vb)
    if n - 1 >= 1
        for k in 1:(n - 1)
            l::I = ipvt[k]
            t = b[l]
            if l !== k
                b[l] = b[k]
                b[k] = t
            end
            @views __muladd!(t, a[(k + 1):end, k], b[(k + 1):end])
        end
    end
    for k in n:-1:1
        b[k] = b[k] / a[k, k]
        @views __muladd!(-b[k], a[1:(k - 1), k], b[1:(k - 1)])
    end
    return recursive_unflatten!(vb, b)
end
function __substitute!(a::Matrix{T1}, ipvt::Vector{I}, b::AbstractVector{T2}) where {
        T1, T2 <: Real, I,
    }
    n = size(a, 1)
    if n - 1 >= 1
        for k in 1:(n - 1)
            l::I = ipvt[k]
            t = b[l]
            if l !== k
                b[l] = b[k]
                b[k] = t
            end
            @views __muladd!(t, a[(k + 1):end, k], b[(k + 1):end])
        end
    end
    for k in n:-1:1
        b[k] = b[k] / a[k, k]
        @views __muladd!(-b[k], a[1:(k - 1), k], b[1:(k - 1)])
    end
    return
end

@inline function __muladd!(a, x, y)
    return y .= muladd.(a, x, y)
end

@views function recursive_flatten!(y::Vector, x::Vector{Vector{Vector{T}}}) where {T}
    i = 0
    for xᵢ in x
        for xᵢᵢ in xᵢ
            copyto!(y[(i + 1):(i + length(xᵢᵢ))], xᵢᵢ)
            i += length(xᵢᵢ)
        end
    end
    return nothing
end

@views function recursive_flatten!(y::AbstractArray, x::Vector{Vector{T}}) where {T}
    i = 0
    for xᵢ in x
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    return nothing
end

@views function recursive_flatten!(y::Vector, x::Vector{Vector{T}}) where {
        T <:
        ForwardDiff.Dual,
    }
    i = 0
    for xᵢ in x
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    return nothing
end

@views function recursive_unflatten!(y::Vector{Vector{Vector{T}}}, x::Vector) where {T}
    i = 0
    for yᵢ in y
        for yᵢᵢ in yᵢ
            copyto!(yᵢᵢ, x[(i + 1):(i + length(yᵢᵢ))])
            i += length(yᵢᵢ)
        end
    end
    return nothing
end
@views function recursive_unflatten!(y::Vector{Vector{T}}, x::Vector) where {T}
    i = 0
    for yᵢ in y
        copyto!(yᵢ, x[(i + 1):(i + length(yᵢ))])
        i += length(yᵢ)
    end
    return nothing
end
@views function recursive_unflatten!(y::AbstractArray, x::Vector{T}) where {T}
    i = 0
    for yᵢ in y
        copyto!(yᵢ, x[(i + 1):(i + length(yᵢ))])
        i += length(yᵢ)
    end
    return nothing
end

@inline function construct_bc_jac(prob::BVProblem)
    bcresid_prototype, _ = __get_bcresid_prototype(prob, prob.u0)
    return construct_bc_jac(prob, bcresid_prototype, prob.problem_type)
end
@inline function construct_bc_jac(prob::BVProblem, bcresid_prototype, pt::TwoPointBVProblem)
    return if isinplace(prob)
        bcjac = (df, u, p) -> begin
            _du = similar(u)
            La = length(first(bcresid_prototype))
            @views first(prob.f.bc)(_du[1:La], u, p)
            @views last(prob.f.bc)(_du[(La + 1):end], u, p)
            _f = function (du, u)
                @views first(prob.f.bc)(du[1:La], u, p)
                return @views last(prob.f.bc)(du[(La + 1):end], u, p)
            end
            ForwardDiff.jacobian!(df, _f, _du, u)
            return
        end
    else
        bcjac = (df, u, p) -> begin
            La = length(first(bcresid_prototype))
            _dua = first(prob.f.bc)(u, p)
            _dub = last(prob.f.bc)(u, p)
            _f = function (du, u)
                dua = first(prob.f.bc)(u, p)
                dub = last(prob.f.bc)(u, p)
                return du .= vcat(dua, dub)
            end
            ForwardDiff.jacobian!(df, _f, vcat(_dua, _dub), u)
            return
        end
    end
end
