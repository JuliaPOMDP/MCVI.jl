using MCVI
using Base.Test
import MCVI: init_lower_action
using POMDPModels # for LightDark1d
using POMDPs

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

# @test test_dummy_heuristics()
