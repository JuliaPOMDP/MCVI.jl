using Documenter
using MCVI
using POMDPs
using POMDPModels
using Random

makedocs(
    sitename = "MCVI.jl",
    authors = "Jayesh K. Gupta",
    modules = [MCVI],
    format = Documenter.HTML(),
    # doctest = false,
    checkdocs = :none,
    
)

deploydocs(
           repo   = "github.com/JuliaPOMDP/MCVI.jl.git",
           versions = ["stable" => "v^", "v#.#"]
)
