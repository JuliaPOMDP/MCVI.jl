
mutable struct DummyState end
mutable struct DummyAct end
mutable struct DummyObs end
mutable struct DummyPOMDP <: POMDP{DummyState, DummyAct, DummyObs} end


function test_requirements()
    sim = MCVISimulator()
    solver = MCVISolver(sim, nothing, 1, 100, 8, 500, 1000, 5000, 50,
                        nothing, nothing)
    println("There should be nothing implemented but the default from POMDPs.jl: \n")
    @test_skip @test_throws(MethodError, @requirements_info solver DummyPOMDP())

    solver = MCVISolver(sim, nothing, 1, 100, 8, 500, 1000, 5000, 50,
                        LightDark1DLowerBound(sim.rng), LightDark1DUpperBound(sim.rng))
    println("Everything should be implemented: \n")
    @test_skip @requirements_info solver LightDark1D()
    @show_requirements POMDPs.solve(solver, LightDark1D())
end
