function build_almost_block_diagonals(zeta::Vector{T}, ncomp::I, mesh, ::Type{T}) where {
        T, I,
    }
    lside = 0
    ncol = 2 * ncomp
    n = length(mesh) - 1
    # build integs (describing block structure of matrix)
    rows = Vector{I}(undef, n)
    cols = repeat([ncol], n)
    lasts = repeat([ncomp], n)
    let lside = 0
        for i in 1:(n - 1)
            lside = first(findall(x::Float64 -> x > mesh[i], zeta)) - 1
            rows[i] = ncomp + lside
        end
    end
    lasts[end] = ncol
    rows[end] = ncol
    g = IntermediateAlmostBlockDiagonal([zeros(rows[i], cols[i]) for i in 1:n], lasts)
    return g
end

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

@inline function construct_bc_jac(prob::BVProblem, u0)
    return construct_bc_jac(prob, __get_bcresid_prototype(prob, u0), prob.problem_type)
end
@inline function construct_bc_jac(prob::BVProblem, _, pt::StandardBVProblem)
    if isinplace(prob)
        bcjac = (df, u, p, t) -> begin
            _du = similar(u)
            prob.f.bc(_du, u, p, t)
            _f = @closure (du, u) -> prob.f.bc(du, u, p, t)
            ForwardDiff.jacobian!(df, _f, _du, u)
            return
        end
    else
        bcjac = (df, u, p, t) -> begin
            _du = prob.f.bc(u, p, t)
            _f = @closure (du, u) -> (du .= prob.f.bc(u, p, t))
            ForwardDiff.jacobian!(df, _f, _du, u)
            return
        end
    end
    return bcjac
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

# `u0` as a function of `t`; sampled guesses are linearly interpolated.
function __ascher_guess(u0, p, tspan)
    u0 isa Number && return Returns([u0])
    N = __initial_guess_length(u0)
    if N > 0
        ts = __extract_mesh(u0, tspan[1], tspan[2], N - 1)
        us = __flatten_initial_guess(u0)
        return t -> __ascher_interpolate(ts, us, t)
    end
    u0 isa AbstractArray && return Returns(vec(u0))
    return t -> vec(__initial_guess(u0, p, t))
end

function __ascher_interpolate(ts, us, t)
    length(ts) == 1 && return us[:, 1]
    j = clamp(searchsortedlast(ts, t), 1, length(ts) - 1)
    θ = (t - ts[j]) / (ts[j + 1] - ts[j])
    return @views (1 - θ) .* us[:, j] .+ θ .* us[:, j + 1]
end

# Initial collocation iterate taken from the guess: `z` at the mesh points, and per
# collocation point the derivative of `z` (the slope of the piecewise-linear
# interpolant of the guess) and the value of `y`, which is how `dmz` is read back
# by `approx`.
function __ascher_seed_iterate(u0, p, tspan, mesh, rho, ::Type{T}, ncomp, ny) where {T}
    guess = __ascher_guess(u0, p, tspan)
    ncy = ncomp + ny
    n = length(mesh) - 1
    us = [convert(Vector{T}, guess(t)) for t in mesh]
    z = [u[1:ncomp] for u in us]
    y = [u[(ncomp + 1):ncy] for u in us]
    dmz = [[zeros(T, ncy) for _ in rho] for _ in 1:n]
    for i in 1:n
        h = mesh[i + 1] - mesh[i]
        for (j, ρ) in enumerate(rho)
            @. dmz[i][j][1:ncomp] = (z[i + 1] - z[i]) / h
            dmz[i][j][(ncomp + 1):ncy] .= view(guess(mesh[i] + ρ * h), (ncomp + 1):ncy)
        end
    end
    return z, y, dmz
end
