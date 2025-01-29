using Documenter, DocumenterCitations, DocumenterInterLinks
import DiffEqBase

using BoundaryValueDiffEqCore, BoundaryValueDiffEqMIRK, BoundaryValueDiffEqFIRK,
      BoundaryValueDiffEqMIRKN
using BoundaryValueDiffEqShooting
using BoundaryValueDiffEqAscher
using SciMLBase, DiffEqBase
using BoundaryValueDiffEq
using SimpleBoundaryValueDiffEq

cp(joinpath(@__DIR__, "Manifest.toml"),
    joinpath(@__DIR__, "src/assets/Manifest.toml"); force = true)
cp(joinpath(@__DIR__, "Project.toml"),
    joinpath(@__DIR__, "src/assets/Project.toml"); force = true)

include("pages.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

interlinks = InterLinks("ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "LineSearch" => "https://sciml.github.io/LineSearch.jl/dev/")

makedocs(; sitename = "BoundaryValueDiffEq.jl",
    authors = "SciML",
    modules = [BoundaryValueDiffEqCore, BoundaryValueDiffEqMIRK, BoundaryValueDiffEqFIRK,
        BoundaryValueDiffEqMIRKN, BoundaryValueDiffEqShooting, BoundaryValueDiffEqAscher,
        SciMLBase, DiffEqBase, BoundaryValueDiffEq, SimpleBoundaryValueDiffEq],
    clean = true,
    doctest = false,
    checkdocs = :exports,
    warnonly = [:missing_docs],
    plugins = [bib, interlinks],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/BoundaryValueDiffEq/stable/"),
    pages)

deploydocs(repo = "github.com/SciML/BoundaryValueDiffEq.jl.git"; push_preview = true)
