using BoundaryValueDiffEq
using BoundaryValueDiffEqAscher
using BoundaryValueDiffEqCore
using BoundaryValueDiffEqFIRK
using BoundaryValueDiffEqMIRK
using BoundaryValueDiffEqMIRKN
using BoundaryValueDiffEqShooting
using Test

const PUBLIC_API_SOURCE_DIRS = Dict(
    BoundaryValueDiffEq => ["src"],
    BoundaryValueDiffEqAscher => ["lib/BoundaryValueDiffEqAscher/src"],
    BoundaryValueDiffEqCore => ["lib/BoundaryValueDiffEqCore/src"],
    BoundaryValueDiffEqFIRK => ["lib/BoundaryValueDiffEqFIRK/src"],
    BoundaryValueDiffEqMIRK => ["lib/BoundaryValueDiffEqMIRK/src"],
    BoundaryValueDiffEqMIRKN => ["lib/BoundaryValueDiffEqMIRKN/src"],
    BoundaryValueDiffEqShooting => ["lib/BoundaryValueDiffEqShooting/src"],
)

function strip_comment(line)
    io = IOBuffer()
    in_string = false
    quote_char = '\0'
    escaped = false
    for char in line
        if escaped
            print(io, char)
            escaped = false
        elseif char == '\\'
            print(io, char)
            escaped = true
        elseif in_string
            print(io, char)
            char == quote_char && (in_string = false)
        elseif char == '"' || char == '\''
            print(io, char)
            in_string = true
            quote_char = char
        elseif char == '#'
            break
        else
            print(io, char)
        end
    end
    return String(take!(io))
end

function strip_docstrings(text)
    return replace(text, r"(?s)\"\"\".*?\"\"\"" => "")
end

function marked_names(statement)
    statement = replace(statement, r"^\s*(export|public|@public)\s+" => "")
    statement = replace(statement, ')' => " ")
    statement = replace(statement, '(' => " ")
    names = Symbol[]
    for part in split(statement, ',')
        token = strip(part)
        isempty(token) && continue
        startswith(token, "@") && (token = token[2:end])
        occursin(r"^[A-Za-z_][A-Za-z0-9_!]*$", token) || continue
        push!(names, Symbol(token))
    end
    return names
end

function public_api_names(source_dirs)
    names = Symbol[]
    for source_dir in source_dirs
        for (dir, _, files) in walkdir(source_dir)
            for file in files
                endswith(file, ".jl") || continue
                path = joinpath(dir, file)
                lines = split(strip_docstrings(read(path, String)), '\n')
                i = firstindex(lines)
                while i <= lastindex(lines)
                    raw = strip_comment(lines[i])
                    if occursin(r"^\s*(export|public|@public)\s+", raw)
                        statement = raw
                        while endswith(rstrip(statement), ",") && i < lastindex(lines)
                            i += 1
                            statement *= " " * strip_comment(lines[i])
                        end
                        append!(names, marked_names(statement))
                    end
                    i += 1
                end
            end
        end
    end
    return sort!(unique!(names))
end

function docs_entries()
    entries = Set{String}()
    docs_dir = joinpath("docs", "src")
    for (dir, _, files) in walkdir(docs_dir)
        for file in files
            endswith(file, ".md") || continue
            in_docs_block = false
            for line in eachline(joinpath(dir, file))
                stripped = strip(line)
                if startswith(stripped, "```@docs")
                    in_docs_block = true
                elseif in_docs_block && startswith(stripped, "```")
                    in_docs_block = false
                elseif in_docs_block && !isempty(stripped) && !startswith(stripped, "#")
                    push!(entries, stripped)
                end
            end
        end
    end
    return entries
end

function has_binding_docstring(mod::Module, name::Symbol)
    Docs.hasdoc(mod, name) && return true
    isdefined(mod, name) || return false
    value = getfield(mod, name)
    for owner in keys(PUBLIC_API_SOURCE_DIRS)
        isdefined(owner, name) || continue
        getfield(owner, name) === value || continue
        Docs.hasdoc(owner, name) && return true
    end
    return false
end

function has_rendered_docs_entry(entries, mod::Module, name::Symbol)
    name_string = string(name)
    name_string in entries && return true
    "$(nameof(mod)).$(name_string)" in entries && return true
    isdefined(mod, name) || return false
    value = getfield(mod, name)
    for owner in keys(PUBLIC_API_SOURCE_DIRS)
        isdefined(owner, name) || continue
        getfield(owner, name) === value || continue
        "$(nameof(owner)).$(name_string)" in entries && return true
    end
    return false
end

@testset "Public API documentation coverage" begin
    entries = docs_entries()
    missing_docstrings = String[]
    missing_docs_entries = String[]

    for (mod, source_dirs) in sort(collect(PUBLIC_API_SOURCE_DIRS); by = x -> string(nameof(x[1])))
        for name in public_api_names(source_dirs)
            binding = "$(nameof(mod)).$(name)"
            @test isdefined(mod, name)
            isdefined(mod, name) || continue
            has_binding_docstring(mod, name) || push!(missing_docstrings, binding)
            has_rendered_docs_entry(entries, mod, name) || push!(missing_docs_entries, binding)
        end
    end

    @test isempty(missing_docstrings)
    @test isempty(missing_docs_entries)
    if !isempty(missing_docstrings)
        @info "Public API bindings missing docstrings" missing_docstrings
    end
    if !isempty(missing_docs_entries)
        @info "Public API bindings missing docs @docs entries" missing_docs_entries
    end
end
