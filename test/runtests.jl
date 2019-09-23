using MCVI
using Test
import MCVI: init_lower_action, lower_bound, upper_bound
using POMDPs
using Random
using POMDPModelTools
using POMDPSimulators
using POMDPModels # for LightDark1d

# to get tests to pass on POMDPs 0.7.3 - can be removed once support is dropped for 0.7.3
POMDPs.generate_o(m::LightDark1D, sp, rng::AbstractRNG) = rand(rng, observation(m, sp))

# Bounds
mutable struct LightDark1DLowerBound
    rng::AbstractRNG
end

mutable struct LightDark1DUpperBound
    rng::AbstractRNG
end

function lower_bound(lb::LightDark1DLowerBound, p::LightDark1D, s::LightDark1DState)
    _, _, r = gen(DDNOut(:sp,:o,:r), p, s, init_lower_action(p), lb.rng)
    return r * discount(p)
end

function upper_bound(ub::LightDark1DUpperBound, p::LightDark1D, s::LightDark1DState)
    steps = abs(s.y)/p.step_size + 1
    return p.correct_r*(discount(p)^steps)
end

function init_lower_action(p::LightDark1D)
    return 0 # Worst? This depends on the initial state? XXX
end

include("test_policy.jl")

include("test_updater.jl")

include("test_belief.jl")

@test test_dummy_graph()
@test test_dummy_graph2()

@test test_backup()

@test test_belief()

include("test_solve.jl")
@test test_solve()

include("test_simulation.jl")
test_simulation()

include("test_requirements.jl")
test_requirements()

# @test test_dummy_heuristics()
