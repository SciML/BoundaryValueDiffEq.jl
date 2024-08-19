using ReTestItems

#ReTestItems.runtests(joinpath(@__DIR__, "mirk/"))
#ReTestItems.runtests(joinpath(@__DIR__, "misc/"))
#ReTestItems.runtests(joinpath(@__DIR__, "shooting/"))
ReTestItems.runtests(joinpath(@__DIR__, "firk/expanded/"))
ReTestItems.runtests(joinpath(@__DIR__, "firk/nested/"))
#=
if !Sys.iswindows() && !Sys.isapple()
    # Wrappers like ODEInterface don't support parallel testing
    ReTestItems.runtests(joinpath(@__DIR__, "wrappers/"); nworkers = 0)
end
=#