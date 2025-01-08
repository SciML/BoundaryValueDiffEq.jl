# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "Getting Started with BVP solving in Julia" => "tutorials/getting_started.md",
    "Basics" => Any["basics/bvp_problem.md", "basics/bvp_functions.md",
        "basics/solve.md", "basics/autodiff.md"],
    "Solver Summaries and Recommendations" => Any[
        "solvers/mirk.md", "solvers/firk.md", "solvers/shooting.md", "solvers/mirkn.md",
        "solvers/ascher.md", "solvers/wrappers.md", "solvers/simple_solvers.md"],
    "Wrapped Solver APIs" => Any["api/odeinterface.md"],
    "Development Documentation" => Any["devdocs/internal_interfaces.md"]]
