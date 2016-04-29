function test_backup()
    p = LightDark1D()
    policy = MCVIPolicy(p)
    MCVI.initialize_updater!(policy)
    b0 = MCVI.initial_belief(p, 500)
    # s0 = initial_state(p)
    # sim = MCVISimulator(MersenneTwister(420), s0, 1)

    sim = MCVISimulator()

    n, _ = MCVI.backup(b0, policy, sim, p, 500, 1000, 500)
    MCVI.addnode!(policy.updater, n)
    policy.updater.root = n

    v1 = MCVI.evaluate(policy, sim, p)
    n, _ = MCVI.backup(b0, policy, sim, p, 500, 1000, 500)
    MCVI.addnode!(policy.updater, n)
    policy.updater.root = n

    v2 = MCVI.evaluate(policy, sim, p)
    return (v1 - 0.1 < v2)
end
