using Documenter, DocumenterCitations, DocumenterInterLinks
import DiffEqBase

using BoundaryValueDiffEqCore, BoundaryValueDiffEqMIRK, BoundaryValueDiffEqFIRK,
      BoundaryValueDiffEqMIRKN
using BoundaryValueDiffEqShooting
using BoundaryValueDiffEqAscher
using SciMLBase, DiffEqBase
using BoundaryValueDiffEq

cp(joinpath(@__DIR__, "Manifest.toml"),
    joinpath(@__DIR__, "src/assets/Manifest.toml"); force = true)
cp(joinpath(@__DIR__, "Project.toml"),
    joinpath(@__DIR__, "src/assets/Project.toml"); force = true)

include("pages.jl")

makedocs(; sitename = "BoundaryValueDiffEq.jl",
    authors = "SciML",
    modules = [BoundaryValueDiffEqCore, BoundaryValueDiffEqMIRK, BoundaryValueDiffEqFIRK,
        BoundaryValueDiffEqMIRKN, BoundaryValueDiffEqShooting,
        BoundaryValueDiffEqAscher, SciMLBase, DiffEqBase, BoundaryValueDiffEq],
    clean = true,
    doctest = false,
    checkdocs = :exports,
    warnonly = [:missing_docs],
    plugins = [bib, interlinks],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/BoundaryValueDiffEq/stable/"),
    pages)
