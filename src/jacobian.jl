# The Jacobian matrix has a most block diagonal structure
# [B₀ 0  0 ... ... ... ...
#  L₁ R₁ 0 ... ... ... ...
#  0  L₂ R₂ 0 ... ... ...
#  . . . . . . . . . . . .
#  0 ... ... ... Lₙ₋₁ Rₙ₋₁
#  0 ... ...  0 ... ... Bₙ]
# where Bᵢ = ∂g₁/∂yᵢ, Lᵢ = ∂ϕᵢ/∂yᵢ and Rᵢ ∂ϕᵢ/∂y_{i+1}
# Lᵢ = -I - hᵢ*Σᵣbᵣ*(∂Kᵣ/∂yᵢ)
# Rᵢ = I - hᵢ*Σᵣbᵣ*(∂Kᵣ/∂y_{i+1})
# where ∂Kᵣ/∂yᵢ = J[r,i] ((1-vᵣ)I + hᵢ∑(xᵣⱼ*∂Kⱼ/∂yᵢ, (j, 1, r-1)))
# ∂Kᵣ/∂y_{i+1} = J[r,i+1] (vᵣI + hᵢ∑(xᵣⱼ*∂Kⱼ/∂y_{i+1}, (j, 1, r-1)))
# J[r, i] = ∂f/∂yᵢ at r stage
# This file will compute the Jacobian of the MIRK scheme
# from the tableau and the `f(x,y,dy)`, since the residual
# function itself is not automatic differentiable.

# Computing df/dy
fun_jac!(out, fun!, x, du, yi) = ForwardDiff.jacobian!(out, (du, u)->fun!(x, u, du), du, yi)

# ∂K∂y

# Computing Jacobian of boundary condition
# bc_jac(bc, u, residual) = ForwardDiff.jacobian(bc, residual, u)

# len = (((m - 1) * n) + length(((m-1)*n+1):(m*n+k)))*2n + (((m - 1) * n) + length(((m-1)*n+1):(m*n+k)))*k
# function construct_jacobian_indices!(i_ind, j_ind, n, m, k)
#     range1 = 1:((m - 1) * n)
#     end1   = length(range1)*2n
#     range2 = ((m-1)*n+1):(m*n+k)
#     end2   = end1+length(range2)*2n
#     end3   = end2+length(range1)*k
#     range3 = (m*n+1):(m*n+k)
#     i_ind[1:end1] = repeat(range1, inner=n, outer=2)
#     i_ind[(end1+1):end2] = repeat(range2, inner=n, outer=2)
#     i_ind[(end2+1):end3] = repeat(range1, inner=k)
#     i_ind[(end3+1):end]  = repeat(range2, inner=k)
#
#     end1j = length(range1)*n
#     end2j = end1+length(range2)*n
#     j_ind[1:end1j] = repeat(0:n-1, outer=n*(m-1)) + repeat((0:(m-2))*n, inner=n^2) + 1
#     j_ind[(end1j+1):end1] = j_ind[1:end1j] + n
#     j_ind[(end1+1):end2j] = repeat(1:n, outer=n+k)
#     j_ind[(end2j+1):end2] = j_ind[(end1+1):end2j] + (m-1)*n
#     j_ind[(end2+1):end3]  = repeat(range3, outer=(m-1)*n)
#     j_ind[(end3+1):end]   = repeat(range3, outer=n+k)
# end
