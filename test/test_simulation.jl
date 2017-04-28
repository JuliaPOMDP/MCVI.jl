function test_simulation()
    prob = LightDark1D()
    sim = HistoryRecorder(max_steps=100)

    solver = MCVISolver(MCVISimulator(), nothing, 1, 100, 8, 500, 1000, 5000, 50, LightDark1DLowerBound(sim.rng), LightDark1DUpperBound(sim.rng))
    println("Solving...")
    policy = solve(solver, prob)
    println("...Solved")
    @show updater(policy).root
    reward = simulate(sim, prob, policy)
    println("Reward:", reward)
    return true
end
