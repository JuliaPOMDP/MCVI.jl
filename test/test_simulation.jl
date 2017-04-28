function test_simulation()
    prob = LightDark1D()
    sim = HistoryRecorder(max_steps=100, capture_exception=true)

    solver = MCVISolver(MCVISimulator(), nothing, 1, 10, 8, 50, 100, 500, 10, LightDark1DLowerBound(sim.rng), LightDark1DUpperBound(sim.rng))
    println("Solving...")
    policy = solve(solver, prob)
    println("...Solved")
    up = updater(policy)
    hist = simulate(sim, prob, policy)
    println("Simulation ran for $(n_steps(hist)) steps.")
    io = IOBuffer()
    showerror(io, get(hist.exception))
    println("Was terminated because of: $(takebuf_string(io))")
    println("Reward:", discounted_reward(hist))
    return true
end
