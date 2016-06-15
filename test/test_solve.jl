function test_solve()
    prob = LightDark1D()
    sim = MCVISimulator()

    solver = MCVISolver(sim, nothing, 1, 100, 8, 500, 1000, 5000, 50, LightDark1DLowerBound(sim.rng), LightDark1DUpperBound(sim.rng))
    println("Solving...")
    policy = solve(solver, prob)
    println("...Solved")
    up = updater(policy)
    reward = simulate(sim, prob, policy, up, up.root)
    println("Reward:", reward)
    return true
end

function test_dummy_heuristics()
    prob = LightDark1D()
end
