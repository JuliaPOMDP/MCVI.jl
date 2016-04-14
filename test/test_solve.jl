using Debug

function test_solve()
    prob = LightDark1D()
    sim = MCVISimulator()

    solver = MCVISolver(prob, sim, MersenneTwister(42), 2, 100, 2, 500, 1000, 5000)
    println("Solving...")
    policy = solve(solver, prob)
    println("...Solved")
    up = updater(policy)
    reward = simulate(sim, prob, policy, up, up.root)
    println(reward)
    return true
end

function test_dummy_heuristics()
    prob = LightDark1D()
end
