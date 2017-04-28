using MCVI
using Base.Test
import MCVI: init_lower_action, lower_bound, upper_bound
using POMDPs
using POMDPToolbox

using POMDPModels # for LightDark1d

# Bounds
type LightDark1DLowerBound
    rng::AbstractRNG
end

type LightDark1DUpperBound
    rng::AbstractRNG
end

function lower_bound(lb::LightDark1DLowerBound, p::LightDark1D, s::LightDark1DState)
    _, _, r = generate_sor(p, s, init_lower_action(p), lb.rng)
    return r * discount(p)
end

function upper_bound(ub::LightDark1DUpperBound, p::LightDark1D, s::LightDark1DState)
    steps = abs(s.y)/p.step_size + 1
    return p.correct_r*(discount(p)^steps)
end

function MCVI.init_lower_action(p::LightDark1D)
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
@test test_simulation()

# @test test_dummy_heuristics()
