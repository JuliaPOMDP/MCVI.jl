using MCVI
using Test
import MCVI: init_lower_action, lower_bound, upper_bound
using POMDPs
using Random
using POMDPTools
using POMDPModels # for LightDark1d
using POMDPLinter: @requirements_info, @show_requirements

# Bounds
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
@testset "MCVISimulator" begin
   @test test_simulation()
   mcvi_sim = MCVISimulator()
   @test mcvi_sim.times == 1
   @test mcvi_sim.display == false
   @test mcvi_sim.rng == Random.GLOBAL_RNG
   mcvi_sim = MCVISimulator(MersenneTwister(42))
   @test mcvi_sim.times == 1
   @test mcvi_sim.display == false
   @test mcvi_sim.rng == MersenneTwister(42)
   mcvi_sim = MCVISimulator(rng=MersenneTwister(42), times=10, display=true)
   @test mcvi_sim.times == 10
   @test mcvi_sim.display == true
   @test mcvi_sim.rng == MersenneTwister(42)    
end
test_simulation()

include("test_requirements.jl")
test_requirements()

# @test test_dummy_heuristics()
