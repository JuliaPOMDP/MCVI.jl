function dummy_node(up, nid)
    act = -1
    states = LightDark1DState(0, 0)
    ae = [MCVI.AlphaEdge([1], nid)]
    return MCVI.create_node(up, act, states, ae)
end

function dummy_node2(up, nid1, nid2)
    act = 1
    states = [LightDark1DState(0, 0), LightDark1DState(0, 3), LightDark1DState(0, 4.8), LightDark1DState(0, 5.2)]
    ae = [MCVI.AlphaEdge([-1, -1, 200, 200], nid1), MCVI.AlphaEdge([0, -1, -1, -1], nid2)]
    return MCVI.create_node(up, act, states, ae)
end

function build_dummy_graph(pomdp)
    up = MCVIUpdater(pomdp)
    n0 = MCVI.init_node(up, pomdp)
    MCVI.addnode!(up, n0)
    n1 = dummy_node(up, n0.id)
    MCVI.addnode!(up, n1)
    n2 = dummy_node(up, n1.id)
    MCVI.addnode!(up, n2)
    n3 = dummy_node(up, n2.id)
    MCVI.addnode!(up, n3)
    up.root = n3
    return up
end

function test_dummy_graph()
    p = LightDark1D()
    up = build_dummy_graph(p)
    # s0 = initial_state(p)
    # sim = MCVISimulator(MersenneTwister(420), 1)
    sim = MCVISimulator()
    policy = MCVIPolicy(p, up)

    sumv = 0.0
    for i in 1:1000
        ss = initial_state(p, p.rng)
        sim.init_state = ss
        sumv += MCVI.simulate(sim, p, policy, up, up.root)
    end
    sumv /= 1000
    return abs(sumv - (-3.0)) <= 1
end

function test_dummy_graph2()
    pomdp = LightDark1D()
    up = MCVIUpdater(pomdp)
    n0 = MCVI.init_node(up, pomdp)
    MCVI.addnode!(up, n0)
    n1 = dummy_node(up, n0.id)
    MCVI.addnode!(up, n1)
    n2 = dummy_node(up, n1.id)
    MCVI.addnode!(up, n2)
    n3 = dummy_node(up, n2.id)
    MCVI.addnode!(up, n3)
    n4 = dummy_node(up, n3.id)
    MCVI.addnode!(up, n4)
    n5 = dummy_node(up, n4.id)
    MCVI.addnode!(up, n5)
    # Move lefts 5 steps
    rn0 = dummy_node2(up, n5.id, n0.id)
    MCVI.addnode!(up, rn0)
    rn1 = dummy_node2(up, n5.id, rn0.id)
    MCVI.addnode!(up, rn1)
    rn2 = dummy_node2(up, n5.id, rn1.id)
    MCVI.addnode!(up, rn2)
    up.root = rn2

    # s0 = initial_state(pomdp)
    # sim = MCVISimulator(MersenneTwister(420), s0, up.root, 1)
    sim = MCVISimulator()
    policy = MCVIPolicy(pomdp, up)
    dump_json(policy, "/tmp/test_policy.json")
    sumv = 0.0
    for i in 1:1000
        ss = initial_state(pomdp, pomdp.rng)
        sim.init_state = ss
        sumv += MCVI.simulate(sim, pomdp, policy, up, up.root)
    end
    sumv /= 1000
    println(sumv)
    return abs(sumv - (-1)) <= 1
end
