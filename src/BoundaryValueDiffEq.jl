module BoundaryValueDiffEq

using LinearAlgebra, Reexport, Setfield, SparseArrays
@reexport using ADTypes, DiffEqBase, NonlinearSolve

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors
import BandedMatrices: BandedMatrix
import DiffEqBase: solve
import FiniteDiff
import FiniteDiff: JacobianCache, finite_difference_jacobian!
import ForwardDiff
import TruncatedStacktraces: @truncate_stacktrace
import UnPack: @unpack

include("types.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("mirk_tableaus.jl")
include("cache.jl")
include("collocation.jl")
include("jacobian.jl")
include("solve.jl")
include("adaptivity.jl")

export Shooting
export MIRK3, MIRK4, MIRK5, MIRK6
export AutoMultiModeDifferentiation, AutoFastDifferentiation, AutoSparseFastDifferentiation

end
