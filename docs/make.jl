using Documenter, SciMLBase, DiffEqBase, BoundaryValueDiffEq

# cp(joinpath(@__DIR__, "Manifest.toml"), joinpath(@__DIR__, "src/assets/Manifest.toml"),
#     force = true)
# cp(joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "src/assets/Project.toml"),
#     force = true)

include("pages.jl")

makedocs(; sitename = "BoundaryValueDiffEq.jl", authors = "Avik Pal et. al.",
    modules = [BoundaryValueDiffEq, DiffEqBase, SciMLBase],
    clean = true, doctest = true, linkcheck = true,
    linkcheck_ignore = [],
    warnonly = [:cross_references], checkdocs = :export,
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/BoundaryValueDiffEq/stable/"),
    pages)

deploydocs(repo = "github.com/SciML/BoundaryValueDiffEq.jl.git"; push_preview = true)
