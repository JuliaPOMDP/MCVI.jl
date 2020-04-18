# MCVI

[![Build Status](https://travis-ci.org/JuliaPOMDP/MCVI.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/MCVI.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/MCVI.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/MCVI.jl?branch=master)

The Monte Carlo Value Iteration (MCVI) offline solver for `POMDPs.jl`.

Described in

Bai, H., Hsu, D., Lee, W. S., & Ngo, V. A. (2010). Monte Carlo value iteration for continuous-state POMDPs. In Algorithmic foundations of robotics IX (pp. 175-191). Springer, Berlin, Heidelberg.

## Installation

```julia
using POMDPs
POMDPs.add_registry()
import Pkg
Pkg.add("MCVI")
```

## Documentation

See [here](http://juliapomdp.github.io/MCVI.jl/)
