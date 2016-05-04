type MCVISimulator <: POMDPs.Simulator
    rng::AbstractRNG
    init_state                  # In case we want to fix starting state
    times
end
MCVISimulator() = MCVISimulator(MersenneTwister(420), nothing, 1)

function simulate(sim::MCVISimulator, pomdp::POMDPs.POMDP, policy::MCVIPolicy, updater::MCVIUpdater, initial_node::MCVINode)
    sum_reward::Reward = 0
    if sim.init_state != nothing
        s = sim.init_state
    else
        s = initial_state(pomdp, sim.rng)
    end
    for i in 1:sim.times
        n = copy(initial_node)
        disc::Float64 = 1
        sumr::Reward = 0
        while true
            sprime, r = generate_sr(pomdp, s, n.act, sim.rng)
            disc *= discount(pomdp)
            sumr += r*disc
            s = sprime
            if !hasnext(n)
                break
            end
            obs = generate_o(pomdp, nothing, nothing, s, sim.rng)
            n = update(updater, n, n.act, obs)
        end
        sum_reward += sumr
    end
    sum_reward /= sim.times

    if isapprox(sum_reward, 0)
        warn("Simulated reward close to 0")
    end
    return sum_reward
end

function simulate(sim::MCVISimulator, pomdp::POMDPs.POMDP, policy::POMDPs.Policy, updater::POMDPs.Updater, initial_belief::MCVIBelief)
    sum_reward::Reward = 0
    if sim.init_state != nothing
        s = sim.init_state
    else
        s = initial_state(pomdp, sim.rng)
    end
    for i in 1:sim.times
        b = initial_belief
        disc::Float64 = 1
        sumr::Reward = 0
        while true
            a = action(policy, b)
            if isterminal(pomdp, a)
                break
            end
            sprime, r = generate_sr(pomdp, s, a, sim.rng)
            disc *= discount(pomdp)
            sumr += r*disc
            s = sprime
            obs = generate_o(pomdp, nothing, nothing, s, sim.rng)
            b = next(b, a, pomdp, sim.rng)
            b = next(b, obs, pomdp)
        end
        sum_reward += sumr
    end
    sum_reward /= sim.times

    if isapprox(sum_reward, 0)
        warn("Simulated reward close to 0")
    end
    return sum_reward
end
