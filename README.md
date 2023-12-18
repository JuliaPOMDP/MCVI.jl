# MCVI

[![CI](https://github.com/JuliaPOMDP/MCVI.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/MCVI.jl/actions/workflows/CI.yml)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/MCVI.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/MCVI.jl?branch=master)

The Monte Carlo Value Iteration (MCVI) offline solver for `POMDPs.jl`.

Described in

Bai, H., Hsu, D., & Lee, W. S. (2014). Integrated perception and planning in the continuous space: A POMDP approach. *The International Journal of Robotics Research*, 33(9), 1288-1302.

## Installation

```julia
using POMDPs
POMDPs.add_registry()
import Pkg
Pkg.add("MCVI")
```

## Documentation

See [here](http://juliapomdp.github.io/MCVI.jl/)
