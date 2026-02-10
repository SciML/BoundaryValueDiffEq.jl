using SciMLLogging: @verbosity_specifier, SciMLLogging

@verbosity_specifier BVPVerbosity begin
    toggles = (
        :bvpsol_convergence,
        :bvpsol_integrator,
        :bvpsol_linear_solver,
        :bvpsol_resources,
        :bvpsol_bc_inconsistent,
        :colnew_convergence,
        :colnew_matrix,
        :colnew_resources,
        :colnew_input,
        :shooting_initial_guess,
        :multiple_shooting_initial_guess,
        :type_inference,
        :initialization,
        :adaptivity,
        :convergence_result,
        :deprecations,
    )

    presets = (
        None = (
            bvpsol_convergence = Silent(),
            bvpsol_integrator = Silent(),
            bvpsol_linear_solver = Silent(),
            bvpsol_resources = Silent(),
            bvpsol_bc_inconsistent = Silent(),
            colnew_convergence = Silent(),
            colnew_matrix = Silent(),
            colnew_resources = Silent(),
            colnew_input = Silent(),
            shooting_initial_guess = Silent(),
            multiple_shooting_initial_guess = Silent(),
            type_inference = Silent(),
            initialization = Silent(),
            adaptivity = Silent(),
            convergence_result = Silent(),
            deprecations = Silent(),
        ),
        Minimal = (
            bvpsol_convergence = WarnLevel(),
            bvpsol_integrator = WarnLevel(),
            bvpsol_linear_solver = WarnLevel(),
            bvpsol_resources = WarnLevel(),
            bvpsol_bc_inconsistent = WarnLevel(),
            colnew_convergence = WarnLevel(),
            colnew_matrix = WarnLevel(),
            colnew_resources = WarnLevel(),
            colnew_input = WarnLevel(),
            shooting_initial_guess = Silent(),
            multiple_shooting_initial_guess = Silent(),
            type_inference = Silent(),
            initialization = Silent(),
            adaptivity = Silent(),
            convergence_result = Silent(),
            deprecations = WarnLevel(),
        ),
        Standard = (
            bvpsol_convergence = WarnLevel(),
            bvpsol_integrator = WarnLevel(),
            bvpsol_linear_solver = WarnLevel(),
            bvpsol_resources = WarnLevel(),
            bvpsol_bc_inconsistent = WarnLevel(),
            colnew_convergence = WarnLevel(),
            colnew_matrix = WarnLevel(),
            colnew_resources = WarnLevel(),
            colnew_input = WarnLevel(),
            shooting_initial_guess = WarnLevel(),
            multiple_shooting_initial_guess = WarnLevel(),
            type_inference = WarnLevel(),
            initialization = Silent(),
            adaptivity = Silent(),
            convergence_result = Silent(),
            deprecations = WarnLevel(),
        ),
        Detailed = (
            bvpsol_convergence = WarnLevel(),
            bvpsol_integrator = WarnLevel(),
            bvpsol_linear_solver = WarnLevel(),
            bvpsol_resources = WarnLevel(),
            bvpsol_bc_inconsistent = WarnLevel(),
            colnew_convergence = WarnLevel(),
            colnew_matrix = WarnLevel(),
            colnew_resources = WarnLevel(),
            colnew_input = WarnLevel(),
            shooting_initial_guess = WarnLevel(),
            multiple_shooting_initial_guess = WarnLevel(),
            type_inference = WarnLevel(),
            initialization = InfoLevel(),
            adaptivity = Silent(),
            convergence_result = InfoLevel(),
            deprecations = WarnLevel(),
        ),
        All = (
            bvpsol_convergence = InfoLevel(),
            bvpsol_integrator = InfoLevel(),
            bvpsol_linear_solver = InfoLevel(),
            bvpsol_resources = InfoLevel(),
            bvpsol_bc_inconsistent = InfoLevel(),
            colnew_convergence = InfoLevel(),
            colnew_matrix = InfoLevel(),
            colnew_resources = InfoLevel(),
            colnew_input = InfoLevel(),
            shooting_initial_guess = InfoLevel(),
            multiple_shooting_initial_guess = InfoLevel(),
            type_inference = InfoLevel(),
            initialization = InfoLevel(),
            adaptivity = InfoLevel(),
            convergence_result = InfoLevel(),
            deprecations = WarnLevel(),
        ),
    )

    groups = (
        bvpsol = (:bvpsol_convergence, :bvpsol_integrator, :bvpsol_linear_solver,
                  :bvpsol_resources, :bvpsol_bc_inconsistent),
        colnew = (:colnew_convergence, :colnew_matrix, :colnew_resources, :colnew_input),
        shooting = (:shooting_initial_guess, :multiple_shooting_initial_guess),
        convergence = (:bvpsol_convergence, :colnew_convergence),
        linear_algebra = (:bvpsol_linear_solver, :colnew_matrix),
        resources = (:bvpsol_resources, :colnew_resources),
        input_validation = (:bvpsol_bc_inconsistent, :colnew_input, :shooting_initial_guess,
                           :multiple_shooting_initial_guess, :type_inference),
        solver_failures = (:bvpsol_convergence, :bvpsol_integrator, :bvpsol_linear_solver,
                          :bvpsol_resources, :bvpsol_bc_inconsistent, :colnew_convergence,
                          :colnew_matrix, :colnew_resources, :colnew_input),
        progress = (:initialization, :adaptivity, :convergence_result),
    )
end

"""
    BVPVerbosity

Verbosity specifier for controlling logging output in BoundaryValueDiffEq.jl.

# Toggles

Fine-grained toggles for specific message categories:

**BVPSOL Solver:**
- `:bvpsol_convergence` - Convergence failures (Gauss Newton, iterative refinement)
- `:bvpsol_integrator` - Integrator trajectory failures
- `:bvpsol_linear_solver` - Linear solver failures (sparse solver, condensing algorithm, rank reduction)
- `:bvpsol_resources` - Resource exhaustion (workspace limits)
- `:bvpsol_bc_inconsistent` - Boundary condition inconsistencies

**COLNEW Solver:**
- `:colnew_convergence` - Nonlinear iteration convergence failures
- `:colnew_matrix` - Collocation matrix singularity
- `:colnew_resources` - Subinterval storage exhaustion
- `:colnew_input` - Input data errors

**Shooting Methods:**
- `:shooting_initial_guess` - Single shooting initial guess warnings
- `:multiple_shooting_initial_guess` - Multiple shooting initial guess warnings

**General:**
- `:type_inference` - Type stability warnings
- `:initialization` - ODE solver initialization failures
- `:adaptivity` - Mesh refinement and adaptivity messages
- `:convergence_result` - Final solve status with residuals
- `:deprecations` - Deprecation warnings

# Presets

BVPVerbosity supports five predefined SciMLLogging presets:

- **None()** - No output (best for production/batch operations)
- **Minimal()** - Only solver failures and deprecations
- **Standard()** - Adds input validation warnings (default, recommended)
- **Detailed()** - Adds initialization and convergence info
- **All()** - Maximum verbosity (includes mesh adaptivity)

# Usage

## Using Presets

```julia
# Standard preset (default)
solve(prob, MIRK4(); verbose = BVPVerbosity())
solve(prob, MIRK4(); verbose = BVPVerbosity(Standard()))

# Completely silent
solve(prob, MIRK4(); verbose = BVPVerbosity(None()))

# Only solver failures
solve(prob, MIRK4(); verbose = BVPVerbosity(Minimal()))

# Detailed debugging
solve(prob, MIRK4(); verbose = BVPVerbosity(Detailed()))

# Maximum verbosity
solve(prob, MIRK4(); verbose = BVPVerbosity(All()))
```

## Setting Individual Toggles

```julia
# Show only BVPSOL convergence issues
solve(prob, BVPSOL(); verbose = BVPVerbosity(
    bvpsol_convergence = WarnLevel(),
    # All others default to Silent in None preset
))

# Silence initial guess warnings but keep solver failures
solve(prob, MultipleShooting(10); verbose = BVPVerbosity(
    Standard(),
    multiple_shooting_initial_guess = Silent()
))

# Only show linear algebra issues
solve(prob, BVPSOL(); verbose = BVPVerbosity(
    None(),
    bvpsol_linear_solver = WarnLevel(),
    colnew_matrix = WarnLevel()
))
```

## Using Groups

Groups provide convenient access to related toggles:

```julia
# All BVPSOL messages
solve(prob, BVPSOL(); verbose = BVPVerbosity(bvpsol = WarnLevel()))

# All COLNEW messages
solve(prob, COLNEW(); verbose = BVPVerbosity(colnew = WarnLevel()))

# All shooting method warnings
solve(prob, Shooting(); verbose = BVPVerbosity(shooting = WarnLevel()))

# All convergence-related failures
solve(prob, alg; verbose = BVPVerbosity(convergence = WarnLevel()))

# All linear algebra issues
solve(prob, alg; verbose = BVPVerbosity(linear_algebra = WarnLevel()))

# All resource exhaustion issues
solve(prob, alg; verbose = BVPVerbosity(resources = WarnLevel()))

# All input validation warnings
solve(prob, alg; verbose = BVPVerbosity(input_validation = WarnLevel()))

# All solver failures (equivalent to Minimal preset failures)
solve(prob, alg; verbose = BVPVerbosity(solver_failures = WarnLevel()))
```

## Backward Compatibility

```julia
# Boolean verbose still works (true → Standard, false → None)
solve(prob, MIRK4(); verbose = true)   # Uses Standard preset
solve(prob, MIRK4(); verbose = false)  # Uses None preset
```

# Groups

Groups for convenient toggle control:

- **`bvpsol`** - All BVPSOL toggles
- **`colnew`** - All COLNEW toggles
- **`shooting`** - All shooting method toggles
- **`convergence`** - Convergence failures across solvers
- **`linear_algebra`** - Linear algebra failures
- **`resources`** - Resource exhaustion issues
- **`input_validation`** - Input validation warnings
- **`solver_failures`** - All solver failure toggles
- **`progress`** - Initialization, adaptivity, and convergence result

# Preset Details

## None
- All toggles: `Silent()`

## Minimal
- All solver failure toggles → `WarnLevel()`
- `deprecations` → `WarnLevel()`
- All others → `Silent()`

## Standard (Default)
- All solver failure toggles → `WarnLevel()`
- All shooting and input validation toggles → `WarnLevel()`
- All others → `Silent()`

## Detailed
- All solver failure toggles → `WarnLevel()`
- All shooting and input validation toggles → `WarnLevel()`
- `initialization` → `InfoLevel()`
- `convergence_result` → `InfoLevel()`
- `adaptivity` → `Silent()` (still off, can be very verbose)

## All
- All toggles → `InfoLevel()` (including adaptivity)
- `deprecations` → `WarnLevel()`
"""
function BVPVerbosity end

const DEFAULT_VERBOSE = BVPVerbosity()

@inline function _process_verbose_param(verbose::SciMLLogging.AbstractVerbosityPreset)
    return BVPVerbosity(verbose)
end

@inline function _process_verbose_param(verbose::Bool)
    return verbose ? DEFAULT_VERBOSE : BVPVerbosity(SciMLLogging.None())
end

@inline _process_verbose_param(verbose::BVPVerbosity) = verbose
