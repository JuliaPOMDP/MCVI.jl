using Documenter, MCVI

makedocs(modules=MCVI)

deploydocs(
           repo   = "github.com/JuliaPOMDP/MCVI.jl.git",
           julia  = "0.5",
           osname = "linux"
           )
