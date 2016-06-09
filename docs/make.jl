using Documenter, MCVI

makedocs(modules=MCVI)

deploydocs(
           repo   = "github.com/JuliaPOMDP/MCVI.jl.git",
           julia  = "release",
           osname = "linux"
           )
