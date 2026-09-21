using BoundaryValueDiffEqShooting, OrdinaryDiffEqTsit5
using ForwardDiff, LinearAlgebra, SparseArrays, StaticArrays, Test

const DeviceShooting = BoundaryValueDiffEqShooting

function device_oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end
device_oscillator(u, p, t) = SVector(u[2], -u[1])
device_left!(r, u, p) = (r[1] = u[1]; nothing)
device_right!(r, u, p) = (r[1] = u[1] - sin(one(eltype(u))); nothing)
device_left(u, p) = SVector(u[1])
device_right(u, p) = SVector(u[1] - sin(one(eltype(u))))

function device_shooting_tests(platform; gpu = false)
    return @testset "Resident shooting $(typeof(platform))" begin
        @testset "Endpoints, interpolation, and OOP, $T" for T in (Float32, Float64)
            tol = T === Float32 ? 2.0e-5 : 1.0e-9
            for iip in (true, false)
                rhs = iip ? device_oscillator! : device_oscillator
                bc = iip ? (device_left!, device_right!) : (device_left, device_right)
                prob = TwoPointBVProblem(rhs, bc, T[0.1, 0.9], (T(0), T(1)); bcresid_prototype = (zeros(T, 1), zeros(T, 1)))
                alg = MultipleShooting(8, Tsit5(); platform, device_steps = 4)
                sol = solve(prob, alg; abstol = tol, nlsolve_kwargs = (; abstol = tol, reltol = tol / 10))
                @test successful_retcode(sol)
                @test maximum(abs, sol.resid) < 5tol
                @test Array(sol.u[end]) ≈ T[sin(1), cos(1)] atol = 5tol
                @test Array(sol(T(0.5))) ≈ T[sin(0.5), cos(0.5)] atol = 5tol
                @test Array(sol(T(0.37))) ≈ T[sin(0.37), cos(0.37)] atol = max(5tol, 1.0e-6)
                interpolated = similar(sol.u[1])
                sol(interpolated, T(0.37))
                @test Array(interpolated) ≈ T[sin(0.37), cos(0.37)] atol = max(5tol, 1.0e-6)
                @test sol(T(0.5); idxs = 1) ≈ sin(0.5) atol = 5tol
                @test Array(sol(T(0.5), Val{1})) ≈ T[cos(0.5), -sin(0.5)] atol = 5tol
                @test_throws BoundsError sol(0.5; idxs = 3)
                @test_throws ArgumentError sol(0.5, Val{2})
                @test length(sol.t) == 9
                @test Array(sol(T[0, 0.5, 1]).u[2]) ≈ T[sin(0.5), cos(0.5)] atol = 5tol
                gpu && @test parent(sol.u[1]) isa CUDA.CuArray
            end
        end

        @testset "Sparse AD and finite differences" begin
            prob = TwoPointBVProblem(device_oscillator!, (device_left!, device_right!), [0.1, 0.9], (0.0, 1.0); bcresid_prototype = (zeros(1), zeros(1)))
            for mode in (AutoSparse(AutoForwardDiff(; chunksize = 1)), AutoSparse(AutoForwardDiff(; chunksize = 2)), AutoSparse(AutoFiniteDiff()), AutoSparse(AutoFiniteDiff(; fdjtype = Val(:central))))
                alg = MultipleShooting(5, Tsit5(); platform, device_steps = 8, jac_alg = BVPJacobianAlgorithm(mode))
                setup = DeviceShooting.__shooting_device_setup(prob, alg)
                (; u, cache, work, plan) = setup
                DeviceShooting.__shooting_jacobian!(plan.matrix, u, cache, plan)
                J = Matrix(gpu ? SparseMatrixCSC(plan.matrix) : plan.matrix)
                expected = zeros(12, 12)
                expected[1, 1] = 1
                expected[end, end - 1] = 1
                flow = [cos(0.2) sin(0.2); -sin(0.2) cos(0.2)]
                for i in 1:5
                    expected[(2i):(2i + 1), (2i - 1):(2i)] = flow
                    expected[(2i):(2i + 1), (2i + 1):(2i + 2)] = -Matrix{Float64}(I, 2, 2)
                end
                @test J ≈ expected atol = mode == AutoSparse(AutoFiniteDiff()) ? 1.0e-7 : 1.0e-8
                @test nnz(plan.matrix) <= 6 * 5 + 4
                @test plan.groups[1].ncolors <= 4
                gpu && @test !(plan.matrix isa SparseMatrixCSC)
                # Repeated assembly overwrites every stored value.
                fill!(nonzeros(plan.matrix), NaN)
                DeviceShooting.__shooting_jacobian!(plan.matrix, u, cache, plan)
                @test all(isfinite, nonzeros(plan.matrix))
            end
        end

        @testset "Nonlinear multipoint BC and compressed derivatives" begin
            function rhs!(du, u, p, t)
                du[1] = u[2]
                du[2] = -sin(t) + p[1] * (u[1]^3 - sin(t)^3)
                return nothing
            end
            function bc!(r, sol, p, t)
                r[1] = sol(0.0)[1]
                r[2] = sol(0.37)[1]^2 - sin(0.37)^2
                return nothing
            end
            prob = BVProblem(rhs!, bc!, (p, t) -> [sin(t) + 0.03, cos(t) - 0.02], (0.0, 1.0), [0.1])
            alg = MultipleShooting(16, Tsit5(); platform, device_steps = 4)
            sol = solve(prob, alg; abstol = 1.0e-10)
            @test successful_retcode(sol)
            @test Array(sol.u[end]) ≈ [sin(1), cos(1)] atol = 2.0e-6
            @test abs(sol(0.37; idxs = 1)^2 - sin(0.37)^2) < 1.0e-9
            setup = DeviceShooting.__shooting_device_setup(prob, alg)
            (; u, cache, work, plan) = setup
            DeviceShooting.__shooting_jacobian!(plan.matrix, u, cache, plan)
            J = Matrix(gpu ? SparseMatrixCSC(plan.matrix) : plan.matrix)
            # Independent dense central differences of the full nonlinear residual.
            reference = zeros(size(J))
            host_u = Array(u)
            h = 1.0e-5
            for j in eachindex(host_u)
                plus, minus = copy(host_u), copy(host_u)
                plus[j] += h; minus[j] -= h
                rp, rm = similar(work.residual), similar(work.residual)
                DeviceShooting.__shooting_residual!(rp, DeviceShooting.__shooting_copy(platform, plus), cache, work)
                DeviceShooting.__shooting_residual!(rm, DeviceShooting.__shooting_copy(platform, minus), cache, work)
                reference[:, j] = (Array(rp) - Array(rm)) / (2h)
            end
            @test J ≈ reference atol = 1.0e-8
            @test plan.groups[2].ncolors <= 4
        end

        @testset "Transitive sparsity, time branches, and multiple AD chunks" begin
            function coupled!(du, u, p, t)
                total = zero(eltype(u))
                for j in eachindex(u)
                    total += u[j] / length(u)
                end
                for j in eachindex(u)
                    du[j] = u[j] + total
                end
                return nothing
            end
            # Explicit loops also work inside GPU kernels.
            function left_loop!(r, u, p)
                for j in eachindex(r)
                    r[j] = u[j] - 1
                end
                return nothing
            end
            empty_bc!(r, u, p) = nothing
            n = 10
            prob = TwoPointBVProblem(
                coupled!, (left_loop!, empty_bc!), ones(n), (0.0, 1.0);
                bcresid_prototype = (zeros(n), zeros(0))
            )
            alg = MultipleShooting(4, Tsit5(); platform, device_steps = 8)
            (; u, cache, work, plan) = DeviceShooting.__shooting_device_setup(prob, alg)
            DeviceShooting.__shooting_jacobian!(plan.matrix, u, cache, plan)
            J = Matrix(gpu ? SparseMatrixCSC(plan.matrix) : plan.matrix)
            flow = exp(0.25) * (Matrix{Float64}(I, n, n) + (exp(0.25) - 1) / n * ones(n, n))
            expected = zeros(size(J))
            expected[1:n, 1:n] = Matrix{Float64}(I, n, n)
            for i in 1:4
                expected[(i * n + 1):((i + 1) * n), ((i - 1) * n + 1):(i * n)] = flow
                expected[(i * n + 1):((i + 1) * n), (i * n + 1):((i + 1) * n)] = -Matrix{Float64}(I, n, n)
            end
            @test plan.groups[1].ncolors > 8
            @test J ≈ expected atol = 1.0e-9
            function time_branch!(du, u, p, t)
                du[1] = t < 0.5 ? u[1] : u[2]
                du[2] = u[2]
                return nothing
            end
            branchprob = TwoPointBVProblem(
                time_branch!, (device_left!, device_right!), ones(2), (0.0, 1.0);
                bcresid_prototype = (zeros(1), zeros(1))
            )
            @test nnz(DeviceShooting.__shooting_flow_pattern(branchprob, 2, Float64)) == 4
            function chain!(du, u, p, t)
                du[1] = u[2]
                du[2] = u[3]
                du[3] = zero(eltype(u))
                return nothing
            end
            chainprob = TwoPointBVProblem(
                chain!, (left_loop!, empty_bc!), ones(3), (0.0, 1.0);
                bcresid_prototype = (zeros(3), zeros(0))
            )
            @test Matrix(DeviceShooting.__shooting_flow_pattern(chainprob, 3, Float64)) == Bool[1 1 1; 0 1 1; 0 0 1]
        end

        @testset "Failure propagation and least squares" begin
            function inconsistent!(r, u, p)
                r[1] = u[1] - 1
                r[2] = u[1] - 2
                return nothing
            end
            prob = TwoPointBVProblem(device_oscillator!, (inconsistent!, device_right!), [0.1, 0.9], (0.0, 1.0); bcresid_prototype = (zeros(2), zeros(1)), nlls = Val(true))
            sol = solve(prob, MultipleShooting(4, Tsit5(); platform, device_steps = 4); nlsolve_kwargs = (; maxiters = 50, abstol = 1.0e-8))
            @test all(isfinite, sol.u[end])
            @test Array(sol.u[1])[1] ≈ 1.5 atol = 1.0e-5
            square = TwoPointBVProblem(device_oscillator!, (device_left!, device_right!), [0.1, 0.9], (0.0, 1.0); bcresid_prototype = (zeros(1), zeros(1)))
            failed = solve(square, MultipleShooting(4, Tsit5(); platform, device_steps = 4); nlsolve_kwargs = (; maxiters = 0))
            @test !successful_retcode(failed)
            @test failed.retcode == failed.original.retcode
            @test_throws ArgumentError solve(square, MultipleShooting(4, Tsit5(); platform, device_steps = 2, grid_coarsening = true))
            @test_throws ArgumentError solve(square, MultipleShooting(4, Tsit5(); platform, device_steps = 2); odesolve_kwargs = (; adaptive = true))
            @test_throws ArgumentError MultipleShooting(4, Tsit5(); device_steps = 0)
        end
    end
end
