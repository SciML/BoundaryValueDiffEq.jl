using BoundaryValueDiffEqFIRK, Test
using ForwardDiff, SparseArrays
const FIRK = BoundaryValueDiffEqFIRK

function device_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * u[1]
    return nothing
end
device_rhs(u, p, t) = (u[2], p[1] * u[1])
function device_bc!(r, u, p, t)
    r[1] = u(t[1])[1] - 1
    r[2] = u(t[end])[1] - exp(t[end])
    return nothing
end
device_bc(u, p, t) = (u(t[1])[1] - 1, u(t[end])[1] - exp(t[end]))
device_bca!(r, u, p) = (r[1] = u[1] - 1; nothing)
device_bcb!(r, u, p) = (r[1] = u[1] - exp(one(eltype(u))); nothing)
device_bca(u, p) = (u[1] - 1,)
device_bcb(u, p) = (u[1] - exp(one(eltype(u))),)
function device_interior_bc!(r, u, p, t)
    r[1] = u(oftype(t[1], 0.13))[1] - exp(oftype(t[1], 0.13))
    r[2] = u(oftype(t[1], 0.87))[1] - exp(oftype(t[1], 0.87))
    return nothing
end

function device_problem(upload, T, iip, twopoint; interior = false)
    u0, p = upload(T[0.5, 0.5]), upload(T[1])
    f = iip ? device_rhs! : device_rhs
    if twopoint
        bc = iip ? (device_bca!, device_bcb!) : (device_bca, device_bcb)
        return TwoPointBVProblem(
            f, bc, u0, (zero(T), one(T)), p;
            bcresid_prototype = (upload(zeros(T, 1)), upload(zeros(T, 1))), nlls = Val(false)
        )
    end
    return BVProblem(
        f, interior ? device_interior_bc! : (iip ? device_bc! : device_bc),
        u0, (zero(T), one(T)), p
    )
end

function packed_init(prob, alg; kwargs...)
    return FIRK.__init_firk_device(prob, alg, FIRK.__device_initial_state(prob.u0, prob.p, first(prob.tspan)); kwargs...)
end
