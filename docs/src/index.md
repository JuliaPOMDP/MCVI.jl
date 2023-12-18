# MCVI

[![CI](https://github.com/JuliaPOMDP/MCVI.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/MCVI.jl/actions/workflows/CI.yml)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/MCVI.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/MCVI.jl?branch=master)

The Monte Carlo Value Iteration (MCVI) offline solver for `POMDPs.jl`.

Described in

Bai, H., Hsu, D., & Lee, W. S. (2014). Integrated perception and planning in the continuous space: A POMDP approach. *The International Journal of Robotics Research*, 33(9), 1288-1302.

## Installation

```julia
using Pkg
Pkg.add("MCVI")
```

## Example

```jldoctest
using POMDPs
using POMDPModels
using MCVI
using Random

mutable struct LightDark1DLowerBound
    rng::AbstractRNG
end

mutable struct LightDark1DUpperBound
    rng::AbstractRNG
end

function MCVI.init_lower_action(p::LightDark1D)
    return 0
end

function MCVI.lower_bound(lb::LightDark1DLowerBound, p::LightDark1D, s::LightDark1DState)
    r = @gen(:r)(p, s, MCVI.init_lower_action(p), lb.rng)
    return r * discount(p)
end

function MCVI.upper_bound(ub::LightDark1DUpperBound, p::LightDark1D, s::LightDark1DState)
    steps = abs(s.y)/p.step_size + 1
    return p.correct_r*(discount(p)^steps)
end

prob = LightDark1D()
sim = MCVISimulator(rng=MersenneTwister(1))

solver = MCVISolver(sim, nothing, 1, 100, 8, 500, 1000, 5000, 50, LightDark1DLowerBound(sim.rng), LightDark1DUpperBound(sim.rng))

println("Solving...")
policy = solve(solver, prob)
println("Solved!")

up = updater(policy)
reward = simulate(MCVISimulator(rng=MersenneTwister(1)), prob, policy, up, up.root)
println("Reward: ", reward)

# output
Solving...
Gap closed!
Solved!
Reward: 5.314410000000001
```

## Documentation

```@docs
MCVISolver
```