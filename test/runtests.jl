using MCVI
using Base.Test
import MCVI: init_lower_action
using POMDPModels # for LightDark1d

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
