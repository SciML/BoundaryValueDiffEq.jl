module BoundaryValueDiffEq

using LinearAlgebra, Reexport, Setfield, SparseArrays
@reexport using DiffEqBase, NonlinearSolve

import ArrayInterface: matrix_colors
import BandedMatrices: BandedMatrix
import DiffEqBase: solve
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
export GeneralMIRK3, GeneralMIRK4, GeneralMIRK5, GeneralMIRK6
export MIRK3, MIRK4, MIRK5, MIRK6

end
