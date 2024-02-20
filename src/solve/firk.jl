#= @concrete struct FIRKCache{iip, T} <: AbstractRKCache{iip, T}
	order::Int                 # The order of MIRK method
	stage::Int                 # The state of MIRK method
	M::Int                     # The number of equations
	in_size
	f
	bc
	prob                       # BVProblem
	problem_type               # StandardBVProblem
	p                          # Parameters
	alg                        # FIRK methods
	TU                         # FIRK Tableau
	bcresid_prototype
	# Everything below gets resized in adaptive methods
	mesh                       # Discrete mesh
	mesh_dt                    # Step size
	k_discrete                 # Stage information associated with the discrete Runge-Kutta method
	y
	y₀
	residual
	# The following 2 caches are never resized
	fᵢ_cache
	fᵢ₂_cache
	defect
	kwargs
end
	# FIRK specific
	#nest_cache # cache for the nested nonlinear solve
	#p_nestprob =#

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

    if __needs_diffcache(alg.jac_alg.diffmode) # Test for forward diff
	p_nestprob_cache = Dual{ForwardDiff.Tag{SparseDiffTools.SparseDiffToolsTag, T},
		T, _chunk}.(p_nestprob)

    else
        p_nestprob_cache = copy(p_nestprob)
    end

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
	has_initial_guess, T, M, n, X = __extract_problem_details(prob; dt,
		check_positive_dt = true)
	stage = alg_stage(alg)
	TU, ITU = constructRK(alg, T)

	expanded_jac = isa(TU, FIRKTableau{false})
	chunksize = expanded_jac ? pickchunksize(M + M * n * (stage + 1)) :
				pickchunksize(M * (n + 1))

	__alloc_diffcache = x -> __maybe_allocate_diffcache(vec(x), chunksize, alg.jac_alg)

	fᵢ_cache = __alloc_diffcache(similar(X))
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

	y = __alloc_diffcache.(copy.(y₀))

	k_discrete = [__maybe_allocate_diffcache(similar(X, M, stage), chunksize, alg.jac_alg)
				  for _ in 1:n]

	bcresid_prototype, resid₁_size = __get_bcresid_prototype(prob.problem_type, prob, X)

	residual = if prob.problem_type isa TwoPointBVProblem
		vcat([__alloc_diffcache(__vec(bcresid_prototype))],
			__alloc_diffcache.(copy.(@view(y₀[2:end]))))
	else
		vcat([__alloc_diffcache(bcresid_prototype)],
			__alloc_diffcache.(copy.(@view(y₀[2:end]))))
	end

	defect = [similar(X, ifelse(adaptive, M, 0)) for _ in 1:n]

	# Transform the functions to handle non-vector inputs
	f, bc = if X isa AbstractVector
		prob.f, prob.f.bc
	elseif iip
		vecf!(du, u, p, t) = prob.f(reshape(du, size(X)), reshape(u, size(X)), p, t)
		vecbc! = if !(prob.problem_type isa TwoPointBVProblem)
			function __vecbc!(resid, sol, p, t)
				prob.f.bc(reshape(resid, resid₁_size),
					map(Base.Fix2(reshape, size(X)), sol), p, t)
			end
		else
			function __vecbc_a!(resida, ua, p)
				prob.f.bc[1](reshape(resida, resid₁_size[1]), reshape(ua, size(X)), p)
			end
			function __vecbc_b!(residb, ub, p)
				prob.f.bc[2](reshape(residb, resid₁_size[2]), reshape(ub, size(X)), p)
			end
			(__vecbc_a!, __vecbc_b!)
		end
		bcresid_prototype = vec(bcresid_prototype)
		vecf!, vecbc!
	else
		vecf(u, p, t) = vec(prob.f(reshape(u, size(X)), p, t))
		vecbc = if !(prob.problem_type isa TwoPointBVProblem)
			__vecbc(sol, p, t) = vec(prob.f.bc(map(Base.Fix2(reshape, size(X)), sol), p, t))
		else
			__vecbc_a(ua, p) = vec(prob.f.bc[1](reshape(ua, size(X)), p))
			__vecbc_b(ub, p) = vec(prob.f.bc[2](reshape(ub, size(X)), p))
			(__vecbc_a, __vecbc_b)
		end
		bcresid_prototype = vec(bcresid_prototype)
		vecf, vecbc
	end

	return FIRKCacheExpand{iip, T}(alg_order(alg), stage, M, size(X), f, bc, prob,
		prob.problem_type, prob.p, alg, TU, ITU,
		bcresid_prototype,
		mesh,
		mesh_dt,
		k_discrete, y, y₀, residual, fᵢ_cache,
		fᵢ₂_cache,
		defect,
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

function solve_cache!(nest_cache, _u0, p_nest) # Make reinit! work with forwarddiff
	if eltype(_u0) == Float64
		dual_type = eltype(nest_cache.p)
		#reinit!(nest_cache, u0 = _u0,
		reinit!(nest_cache,
			p = dual_type.(p_nest))
	else
		#reinit!(nest_cache, u0 = _u0, p = p_nest)
		reinit!(nest_cache, p = p_nest)
	end

	return solve!(nest_cache)
end
#= 
function _scalar_nlsolve_∂f_∂p(f, res, u, p)
	return ForwardDiff.jacobian((y, x) -> f(y, u, x), res, p)
end

function _scalar_nlsolve_∂f_∂u(f, res, u, p)
	return ForwardDiff.jacobian((y, x) -> f(y, x, p), res, u)
end

function _scalar_nlsolve_cache_ad(nest_cache::NonlinearSolve.NewtonRaphsonCache{iip}, _u0,
								  p_nest) where {iip}
	_p_nest = ForwardDiff.value.(p_nest)
	new_u0 = ones(size(ForwardDiff.value.(_u0)))

	reinit!(nest_cache, new_u0, p = _p_nest)
	sol = solve!(nest_cache)
	uu = sol.u
	res = zero(uu)

	if iip
		f_p = _scalar_nlsolve_∂f_∂p(nest_cache.f, res, uu, _p_nest)
		f_x = _scalar_nlsolve_∂f_∂u(nest_cache.f, res, uu, _p_nest)
	else
		f_p = NonlinearSolve.scalar_nlsolve_∂f_∂p(nest_cache.f, uu, _p_nest)
		f_x = NonlinearSolve.scalar_nlsolve_∂f_∂u(nest_cache.f, uu, _p_nest)
	end

	z_arr = -inv(f_x) * f_p

	sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
	if uu isa Number
		partials = sum(sumfun, zip(z_arr, p_nest))
	elseif _p_nest isa Number
		partials = sumfun((z_arr, p_nest))
	else
		partials = sum(sumfun, zip(eachcol(z_arr), p_nest))
	end

	return sol, partials
end =#

#TODO: iip overload
#= function solve_cache!(nest_cache, _u0,
					  p_nest::AbstractArray{<:Dual{T, V, P}}) where {T, V, P}

	sol, partials = _scalar_nlsolve_cache_ad(nest_cache, _u0, p_nest);
	dual_soln = map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(sol.u, partials))
	return SciMLBase.build_solution(nest_cache.prob, nest_cache.alg, dual_soln, sol.resid;
									sol.retcode)
end =#

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

# Constructing the Nonlinear Problem
function __construct_nlproblem(cache::FIRKCacheExpand{iip}, y::AbstractVector) where {iip}
	loss_bc = if iip
		function loss_bc_internal!(resid::AbstractVector, u::AbstractVector, p = cache.p)
			y_ = recursive_unflatten!(cache.y, u)
			eval_bc_residual!(resid, cache.problem_type, cache.bc, y_, p, cache.mesh)
			return resid
		end
	else
		function loss_bc_internal(u::AbstractVector, p = cache.p)
			y_ = recursive_unflatten!(cache.y, u)
			return eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
		end
	end

	loss_collocation = if iip
		function loss_collocation_internal!(resid::AbstractVector, u::AbstractVector,
			p = cache.p)
			y_ = recursive_unflatten!(cache.y, u)
			resids = [get_tmp(r, u) for r in cache.residual[2:end]]
			Φ!(resids, cache, y_, u, p)
			recursive_flatten!(resid, resids)
			return resid
		end
	else
		function loss_collocation_internal(u::AbstractVector, p = cache.p)
			y_ = recursive_unflatten!(cache.y, u)
			resids = Φ(cache, y_, u, p)
			return mapreduce(vec, vcat, resids)
		end
	end

	loss = if iip
		function loss_internal!(resid::AbstractVector, u::AbstractVector, p = cache.p)
			y_ = recursive_unflatten!(cache.y, u)
			resids = [get_tmp(r, u) for r in cache.residual]
			eval_bc_residual!(resids[1], cache.problem_type, cache.bc, y_, p, cache.mesh)
			Φ!(resids[2:end], cache, y_, u, p)
			if cache.problem_type isa TwoPointBVProblem
				recursive_flatten_twopoint!(resid, resids)
			else
				recursive_flatten!(resid, resids)
			end
			return resid
		end
	else
		function loss_internal(u::AbstractVector, p = cache.p)
			y_ = recursive_unflatten!(cache.y, u)
			resid_bc = eval_bc_residual(cache.problem_type, cache.bc, y_, p, cache.mesh)
			resid_co = Φ(cache, y_, u, p)
			if cache.problem_type isa TwoPointBVProblem
				return vcat(resid_bc.x[1], mapreduce(vec, vcat, resid_co), resid_bc.x[2])
			else
				return vcat(resid_bc, mapreduce(vec, vcat, resid_co))
			end
		end
	end

	return __construct_nlproblem(cache, y, loss_bc, loss_collocation, loss,
		cache.problem_type)
end

function __construct_nlproblem(cache::FIRKCacheExpand{iip}, y, loss_bc, loss_collocation,
	loss,
	::StandardBVProblem) where {iip}
	@unpack nlsolve, jac_alg = cache.alg
	N = length(cache.mesh)

	TU, ITU = constructRK(cache.alg, eltype(y))

	expanded_jac = isa(TU, FIRKTableau{false})

	resid_bc = cache.bcresid_prototype
	resid_collocation = expanded_jac ? similar(y, cache.M * (N - 1) * (TU.s + 1)) :
						similar(y, cache.M * (N - 1))

	sd_bc = jac_alg.bc_diffmode isa AbstractSparseADType ? SymbolicsSparsityDetection() :
			NoSparsityDetection()
	cache_bc = __sparse_jacobian_cache(Val(iip), jac_alg.bc_diffmode, sd_bc, loss_bc,
		resid_bc, y)

	sd_collocation = if jac_alg.nonbc_diffmode isa AbstractSparseADType
		PrecomputedJacobianColorvec(__generate_sparse_jacobian_prototype(cache,
			cache.problem_type,
			y, cache.M, N, TU))
	else
		NoSparsityDetection()
	end
	cache_collocation = __sparse_jacobian_cache(Val(iip), jac_alg.nonbc_diffmode,
		sd_collocation, loss_collocation,
		resid_collocation, y)

	jac_prototype = vcat(init_jacobian(cache_bc), init_jacobian(cache_collocation))

	jac = if iip
		function jac_internal!(J, x, p)
			sparse_jacobian!(@view(J[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc,
				loss_bc, resid_bc, x)
			sparse_jacobian!(@view(J[(cache.M+1):end, :]), jac_alg.nonbc_diffmode,
				cache_collocation, loss_collocation, resid_collocation, x)
			return J
		end
	else
		J_ = jac_prototype
		function jac_internal(x, p)
			sparse_jacobian!(@view(J_[1:(cache.M), :]), jac_alg.bc_diffmode, cache_bc,
				loss_bc, x)
			sparse_jacobian!(@view(J_[(cache.M+1):end, :]), jac_alg.nonbc_diffmode,
				cache_collocation, loss_collocation, x)
			return J_
		end
	end

	return NonlinearProblem(NonlinearFunction{iip}(loss; jac, jac_prototype), y, cache.p)
end
