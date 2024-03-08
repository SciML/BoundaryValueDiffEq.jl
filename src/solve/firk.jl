@concrete struct FIRKCacheNested{iip, T} <: AbstractRKCache{iip, T}
	order::Int                 # The order of MIRK method
	stage::Int                 # The state of MIRK method
	M::Int                     # The number of equations
	in_size::Any
	f::Any
	bc::Any
	prob::Any                       # BVProblem
	problem_type::Any               # StandardBVProblem
	p::Any                          # Parameters
	alg::Any                        # MIRK methods
	TU::Any                         # MIRK Tableau
	ITU::Any                        # MIRK Interpolation Tableau
	bcresid_prototype::Any
	# Everything below gets resized in adaptive methods
	mesh::Any                       # Discrete mesh
	mesh_dt::Any                    # Step size
	k_discrete::Any                 # Stage information associated with the discrete Runge-Kutta method
	y::Any
	y₀::Any
	residual::Any
	# The following 2 caches are never resized
	fᵢ_cache::Any
	fᵢ₂_cache::Any
	defect::Any
	p_nestprob::Any
	nest_cache::Any
	resid_size::Any
	kwargs::Any
end

@concrete struct FIRKCacheExpand{iip, T} <: AbstractRKCache{iip, T}
	order::Int                 # The order of MIRK method
	stage::Int                 # The state of MIRK method
	M::Int                     # The number of equations
	in_size::Any
	f::Any
	bc::Any
	prob::Any                       # BVProblem
	problem_type::Any               # StandardBVProblem
	p::Any                          # Parameters
	alg::Any                        # MIRK methods
	TU::Any                         # MIRK Tableau
	ITU::Any                        # MIRK Interpolation Tableau
	bcresid_prototype::Any
	# Everything below gets resized in adaptive methods
	mesh::Any                       # Discrete mesh
	mesh_dt::Any                    # Step size
	k_discrete::Any                 # Stage information associated with the discrete Runge-Kutta method
	y::Any
	y₀::Any
	residual::Any
	# The following 2 caches are never resized
	fᵢ_cache::Any
	fᵢ₂_cache::Any
	defect::Any
	resid_size::Any
	kwargs::Any
end

function extend_y(y, N, stage)
	y_extended = similar(y, (N - 1) * (stage + 1) + 1)
	y_extended[1] = y[1]
	let ctr1 = 2
		for i in 2:N
			for j in 1:(stage+1)
				y_extended[(ctr1)] = y[i]
				ctr1 += 1
			end
		end
	end
	return y_extended
end

function shrink_y(y, N, M, stage)
	y_shrink = similar(y, N)
	y_shrink[1] = y[1]
	let ctr = stage + 2
		for i in 2:N
			y_shrink[i] = y[ctr]
			ctr += (stage + 1)
		end
	end
	return y_shrink
end

function SciMLBase.__init(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
	abstol = 1e-3, adaptive = true,
	nlsolve_kwargs = (; abstol = 1e-4, reltol = 1e-4, maxiters = 10),
	kwargs...)
	if alg.nested_nlsolve
		return init_nested(prob, alg; dt = dt,
			abstol = abstol, adaptive = adaptive,
			nlsolve_kwargs = nlsolve_kwargs, kwargs...)
	else
		return init_expanded(prob, alg; dt = dt,
			abstol = abstol, adaptive = adaptive, kwargs...)
	end
end

function init_nested(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
	abstol = 1e-3, adaptive = true, nlsolve_kwargs, kwargs...)
	@set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)
	iip = isinplace(prob)
	if adaptive && isa(alg, FIRKNoAdaptivity)
		error("Algorithm doesn't support adaptivity. Please choose a higher order algorithm.")
	end

	_, T, M, n, X = __extract_problem_details(prob; dt, check_positive_dt = true)
	# NOTE: Assumes the user provided initial guess is on a uniform mesh
	mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
	mesh_dt = diff(mesh)

	chunksize = pickchunksize(M * (n + 1))

	__alloc = x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

	fᵢ_cache = __alloc(similar(X))
	fᵢ₂_cache = vec(similar(X))

	defect_threshold = T(0.1)  # TODO: Allow user to specify these
	MxNsub = 3000              # TODO: Allow user to specify these

	# Don't flatten this here, since we need to expand it later if needed
	y₀ = __initial_state_from_prob(prob, mesh)
	y = __alloc.(copy.(y₀))
	TU, ITU = constructRK(alg, T)
	stage = alg_stage(alg)

	k_discrete = [__maybe_allocate_diffcache(fill(one(T), (M, stage)), chunksize,
		alg.jac_alg)
				  for _ in 1:n]

	bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

	residual = if prob.problem_type isa TwoPointBVProblem
		vcat([__alloc(__vec(bcresid_prototype))], __alloc.(copy.(@view(y₀[2:end]))))
	else
		vcat([__alloc(bcresid_prototype)], __alloc.(copy.(@view(y₀[2:end]))))
	end

	defect = [similar(X, ifelse(adaptive, M, 0)) for _ in 1:n]

	# Transform the functions to handle non-vector inputs
	bcresid_prototype = __vec(bcresid_prototype)
	f, bc = if X isa AbstractVector
		prob.f, prob.f.bc
	elseif iip
		vecf! = (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
		vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
			(r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
		else
			((r, u, p) -> __vec_bc!(r, u, p, prob.f.bc[1], resid₁_size[1], size(X)),
				(r, u, p) -> __vec_bc!(r, u, p, prob.f.bc[2], resid₁_size[2], size(X)))
		end
		vecf!, vecbc!
	else
		vecf = (u, p, t) -> __vec_f(u, p, t, prob.f, size(X))
		vecbc = if !(prob.problem_type isa TwoPointBVProblem)
			(u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(X))
		else
			((u, p) -> __vec_bc(u, p, prob.f.bc[1], size(X))),
			(u, p) -> __vec_bc(u, p, prob.f.bc[2], size(X))
		end
		vecf, vecbc
	end

	prob_ = !(prob.u0 isa AbstractArray) ? remake(prob; u0 = X) : prob

	# Initialize internal nonlinear problem cache
	@unpack c, a, b, s = TU
	p_nestprob = zeros(T, M + 2)

	if isa(u0, AbstractArray) && eltype(prob.u0) <: AbstractVector
		u0_mat = hcat(prob.u0...)
		avg_u0 = vec(sum(u0_mat, dims = 2)) / size(u0_mat, 2)
	else
		avg_u0 = prob.u0
	end

	K0 = repeat(avg_u0, 1, s) # Somewhat arbitrary initialization of K

	if alg.jac_alg.diffmode isa AbstractSparseADType
		_chunk = pickchunksize(length(K0))
	else
		_chunk = chunksize
	end

	p_nestprob_cache = copy(p_nestprob)

	if iip
		nestprob = NonlinearProblem((res, K, p_nestprob) -> FIRK_nlsolve!(res, K,
				p_nestprob, f,
				a, c, stage,
				prob.p),
			K0, p_nestprob_cache)
	else
		nestprob = NonlinearProblem((K, p_nestprob) -> FIRK_nlsolve(K,
				p_nestprob, f,
				a, c, stage,
				prob.p),
			K0, p_nestprob_cache)
	end

	nest_cache = init(nestprob,
		NewtonRaphson(autodiff = alg.jac_alg.diffmode);
		nlsolve_kwargs...)

	return FIRKCacheNested{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob_,
		prob.problem_type, prob.p, alg, TU, ITU,
		bcresid_prototype,
		mesh, mesh_dt,
		k_discrete, y, y₀, residual, fᵢ_cache, fᵢ₂_cache,
		defect, p_nestprob_cache, nest_cache,
		resid₁_size,
		(; defect_threshold, MxNsub, abstol, dt, adaptive,
			kwargs...))
end

function init_expanded(prob::BVProblem, alg::AbstractFIRK; dt = 0.0,
	abstol = 1e-3, adaptive = true,
	nlsolve_kwargs = (; abstol = 1e-3, reltol = 1e-3, maxiters = 10),
	kwargs...)
	@set! alg.jac_alg = concrete_jacobian_algorithm(alg.jac_alg, prob, alg)

	if adaptive && isa(alg, FIRKNoAdaptivity)
		error("Algorithm doesn't support adaptivity. Please choose a higher order algorithm.")
	end

	iip = isinplace(prob)

	_, T, M, n, X = __extract_problem_details(prob; dt,
		check_positive_dt = true)
	stage = alg_stage(alg)
	TU, ITU = constructRK(alg, T)

	expanded_jac = isa(TU, FIRKTableau{false})
	chunksize = expanded_jac ? pickchunksize(M + M * n * (stage + 1)) :
				pickchunksize(M * (n + 1))

	__alloc = x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

	fᵢ_cache = __alloc(similar(X))
	fᵢ₂_cache = vec(similar(X))

	# NOTE: Assumes the user provided initial guess is on a uniform mesh
	mesh = collect(range(prob.tspan[1], stop = prob.tspan[2], length = n + 1))
	mesh_dt = diff(mesh)

	defect_threshold = T(0.1)  # TODO: Allow user to specify these
	MxNsub = 3000              # TODO: Allow user to specify these

	# Don't flatten this here, since we need to expand it later if needed
	y₀ = expanded_jac ?
		 extend_y(__initial_state_from_prob(prob, mesh), n + 1, alg_stage(alg)) :
		 __initial_state_from_prob(prob, mesh)

	y = __alloc.(copy.(y₀))

	k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
				  for _ in 1:n]

	bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

	residual = if prob.problem_type isa TwoPointBVProblem
		vcat([__alloc(__vec(bcresid_prototype))],
			__alloc.(copy.(@view(y₀[2:end]))))
	else
		vcat([__alloc(bcresid_prototype)],
			__alloc.(copy.(@view(y₀[2:end]))))
	end

	defect = [similar(X, ifelse(adaptive, M, 0)) for _ in 1:n]

	# Transform the functions to handle non-vector inputs
	bcresid_prototype = __vec(bcresid_prototype)
	f, bc = if X isa AbstractVector
		prob.f, prob.f.bc
	elseif iip
		vecf! = (du, u, p, t) -> __vec_f!(du, u, p, t, prob.f, size(X))
		vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
			(r, u, p, t) -> __vec_bc!(r, u, p, t, prob.f.bc, resid₁_size, size(X))
		else
			((r, u, p) -> __vec_bc!(r, u, p, prob.f.bc[1], resid₁_size[1], size(X)),
				(r, u, p) -> __vec_bc!(r, u, p, prob.f.bc[2], resid₁_size[2], size(X)))
		end
		vecf!, vecbc!
	else
		vecf = (u, p, t) -> __vec_f(u, p, t, prob.f, size(X))
		vecbc = if !(prob.problem_type isa TwoPointBVProblem)
			(u, p, t) -> __vec_bc(u, p, t, prob.f.bc, size(X))
		else
			((u, p) -> __vec_bc(u, p, prob.f.bc[1], size(X))),
			(u, p) -> __vec_bc(u, p, prob.f.bc[2], size(X))
		end
		vecf, vecbc
	end

	prob_ = !(prob.u0 isa AbstractArray) ? remake(prob; u0 = X) : prob

	return FIRKCacheExpand{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob_,
		prob.problem_type, prob.p, alg, TU, ITU,
		bcresid_prototype,
		mesh,
		mesh_dt,
		k_discrete, y, y₀, residual, fᵢ_cache,
		fᵢ₂_cache,
		defect, resid₁_size,
		(; defect_threshold, MxNsub, abstol, dt, adaptive,
			kwargs...))
end

"""
	__expand_cache!(cache::FIRKCache)

After redistributing or halving the mesh, this function expands the required vectors to
match the length of the new mesh.
"""
function __expand_cache!(cache::Union{FIRKCacheNested, FIRKCacheExpand})
	Nₙ = length(cache.mesh)
	__append_similar!(cache.k_discrete, Nₙ - 1, cache.M, cache.TU)
	__append_similar!(cache.y, Nₙ, cache.M, cache.TU)
	__append_similar!(cache.y₀, Nₙ, cache.M, cache.TU)
	__append_similar!(cache.residual, Nₙ, cache.M, cache.TU)
	__append_similar!(cache.defect, Nₙ - 1, cache.M, cache.TU)
	return cache
end

function solve_cache!(nest_cache, _u0, p_nest) # TODO: Make work with ForwardDiff
	reinit!(nest_cache, p = p_nest)
	return solve!(nest_cache)
end

function SciMLBase.solve!(cache::FIRKCacheExpand)
	(defect_threshold, MxNsub, abstol, adaptive, _), kwargs = __split_mirk_kwargs(;
		cache.kwargs...)
	@unpack y, y₀, prob, alg, mesh, mesh_dt, TU, ITU = cache
	info::ReturnCode.T = ReturnCode.Success
	defect_norm = 2 * abstol

	while SciMLBase.successful_retcode(info) && defect_norm > abstol
		nlprob = __construct_nlproblem(cache, recursive_flatten(y₀))
		sol_nlprob = solve(nlprob, alg.nlsolve; abstol, kwargs...)
		recursive_unflatten!(cache.y₀, sol_nlprob.u)

		info = sol_nlprob.retcode

		!adaptive && break

		if info == ReturnCode.Success
			defect_norm = defect_estimate!(cache)
			# The defect is greater than 10%, the solution is not acceptable
			defect_norm > defect_threshold && (info = ReturnCode.Failure)
		end

		if info == ReturnCode.Success
			if defect_norm > abstol
				# We construct a new mesh to equidistribute the defect
				mesh, mesh_dt, _, info = mesh_selector!(cache)
				if info == ReturnCode.Success
					__append_similar!(cache.y₀, length(cache.mesh), cache.M, cache.TU)
					for (i, m) in enumerate(cache.mesh)
						interp_eval!(cache.y₀, i, cache, cache.ITU, m, mesh, mesh_dt)
					end
					__expand_cache!(cache)
				end
			end
		else
			#  We cannot obtain a solution for the current mesh
			if 2 * (length(cache.mesh) - 1) > MxNsub
				# New mesh would be too large
				info = ReturnCode.Failure
			else
				half_mesh!(cache)
				__expand_cache!(cache)
				recursive_fill!(cache.y₀, 0)
				info = ReturnCode.Success # Force a restart
				defect_norm = 2 * abstol
			end
		end
	end

	# sync y and y0 caches
	for i in axes(cache.y₀, 1)
		cache.y[i].du .= cache.y₀[i]
	end

	u = [reshape(y, cache.in_size) for y in cache.y₀]
	if isa(TU, FIRKTableau{false})
		u = shrink_y(u, length(cache.mesh), cache.M, alg_stage(cache.alg))
	end
	return DiffEqBase.build_solution(prob, alg, cache.mesh,
		u; interp = RKInterpolation(cache.mesh, u, cache),
		retcode = info)
end

function __construct_nlproblem(cache::FIRKCacheExpand{iip}, y, loss_bc::BC, loss_collocation::C,
	loss::LF, ::StandardBVProblem) where {iip, BC, C, LF}
	@unpack nlsolve, jac_alg = cache.alg
	N = length(cache.mesh)
	TU, ITU = constructRK(cache.alg, eltype(y))
	@unpack s = TU

	resid_bc = cache.bcresid_prototype
	L = length(resid_bc)
	resid_collocation = similar(y, cache.M * (N - 1) * (TU.s + 1))

	loss_bcₚ = iip ? ((du, u) -> loss_bc(du, u, cache.p)) : (u -> loss_bc(u, cache.p))
	loss_collocationₚ = iip ? ((du, u) -> loss_collocation(du, u, cache.p)) :
						(u -> loss_collocation(u, cache.p))

	sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
			NoSparsityDetection()
	cache_bc = __sparse_jacobian_cache(Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bcₚ,
		resid_bc, y)

	sd_collocation = if jac_alg.nonbc_diffmode isa AbstractSparseADType
		if L < cache.M
			# For underdetermined problems we use sparse since we don't have banded qr
			colored_matrix = __generate_sparse_jacobian_prototype(cache,
				cache.problem_type, y, y, cache.M, N)
			J_full_band = nothing
			__sparsity_detection_alg(ColoredMatrix(sparse(colored_matrix.M),
				colored_matrix.row_colorvec, colored_matrix.col_colorvec))
		else
			block_size = cache.M * (s + 2)
			J_full_band = BandedMatrix(Ones{eltype(y)}(L + cache.M * (s + 1) * (N - 1), cache.M * (s + 1) * (N - 1) + cache.M),
				(block_size, block_size))
			__sparsity_detection_alg(__generate_sparse_jacobian_prototype(cache,
				cache.problem_type,
				y, cache.M, N, TU))
		end
	else
		J_full_band = nothing
		NoSparsityDetection()
	end

	cache_collocation = __sparse_jacobian_cache(Val(iip), jac_alg.nonbc_diffmode,
		sd_collocation, loss_collocationₚ, resid_collocation, y)

	J_bc = init_jacobian(cache_bc)
	J_c = init_jacobian(cache_collocation)

	if J_full_band === nothing
		jac_prototype = vcat(J_bc, J_c)
	else
		jac_prototype = AlmostBandedMatrix{eltype(cache)}(J_full_band, J_bc)
	end

	jac = if iip
		(J, u, p) -> __mirk_mpoint_jacobian!(J, J_c, u, jac_alg.bc_diffmode,
			jac_alg.nonbc_diffmode, cache_bc, cache_collocation, loss_bcₚ,
			loss_collocationₚ, resid_bc, resid_collocation, L)
	else
		(u, p) -> __mirk_mpoint_jacobian(jac_prototype, J_c, u, jac_alg.bc_diffmode,
			jac_alg.nonbc_diffmode, cache_bc, cache_collocation, loss_bcₚ,
			loss_collocationₚ, L)
	end

	nlf = NonlinearFunction{iip}(loss; resid_prototype = vcat(resid_bc, resid_collocation),
		jac, jac_prototype)
	return (L == cache.M ? NonlinearProblem : NonlinearLeastSquaresProblem)(nlf, y, cache.p)
end


function __construct_nlproblem(cache::FIRKCacheExpand{iip}, y, loss_bc::BC, loss_collocation::C,
	loss::LF, ::TwoPointBVProblem) where {iip, BC, C, LF}
	@unpack nlsolve, jac_alg = cache.alg
	N = length(cache.mesh)

	lossₚ = iip ? ((du, u) -> loss(du, u, cache.p)) : (u -> loss(u, cache.p))

	TU, ITU = constructRK(cache.alg, eltype(y))

	resid_collocation = similar(y, cache.M * (N - 1) * (TU.s + 1))

	resid = vcat(@view(cache.bcresid_prototype[1:prod(cache.resid_size[1])]),
		resid_collocation,
		@view(cache.bcresid_prototype[(prod(cache.resid_size[1])+1):end]))
	L = length(cache.bcresid_prototype)
	
	TU, ITU = constructRK(cache.alg, eltype(y))
	@unpack s = TU
	sd = if jac_alg.nonbc_diffmode isa AbstractSparseADType
		block_size = cache.M * (s + 2)
		J_full_band = BandedMatrix(Ones{eltype(y)}(L + cache.M * (s + 1) * (N - 1), cache.M * (s + 1) * (N - 1) + cache.M),
			(block_size, block_size))
		__sparsity_detection_alg(__generate_sparse_jacobian_prototype(cache,
			cache.problem_type,
			y, cache.M, N, TU))
	else
		J_full_band = nothing
		NoSparsityDetection()
	end
	test = __generate_sparse_jacobian_prototype(cache,
		cache.problem_type,
		y, cache.M, N, TU)
	test.M
	diffcache = __sparse_jacobian_cache(Val(iip), jac_alg.diffmode, sd, lossₚ, resid, y)
	jac_prototype = init_jacobian(diffcache)
	if isdefined(Main, :Infiltrator)
		Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
	end
	jac = if iip
		(J, u, p) -> __mirk_2point_jacobian!(J, u, jac_alg.diffmode, diffcache, lossₚ,
			resid)
	else
		(u, p) -> __mirk_2point_jacobian(u, jac_prototype, jac_alg.diffmode, diffcache,
			lossₚ)
	end

	nlf = NonlinearFunction{iip}(loss; resid_prototype = copy(resid), jac, jac_prototype)

	return (L == cache.M ? NonlinearProblem : NonlinearLeastSquaresProblem)(nlf, y, cache.p)
end
