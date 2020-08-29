mutable struct MCVISimulator <: POMDPs.Simulator
    rng::AbstractRNG
    # init_state                  # In case we want to fix starting state
    times::Integer
    display::Bool
end
MCVISimulator() = MCVISimulator(MersenneTwister(420), 1, false)

function simulate(sim::MCVISimulator, pomdp::POMDPs.POMDP, policy::MCVIPolicy, updater::MCVIUpdater, initial_node::MCVINode, init_state=nothing)
    sum_reward::Reward = 0
    if init_state != nothing
        s = init_state
    else
        s = rand(sim.rng, initialstate(pomdp))
    end
    for i in 1:sim.times
        n = copy(initial_node)
        disc::Float64 = 1
        sumr::Reward = 0
        while true
            a = action(policy, n)
            sprime, obs, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)
            o = @sprintf("%0.6f", obs)
            rtxt = @sprintf("%0.3f", r)
            sim.display && println("s: $s \t a: $a \t sp: $sprime \t o: $o \t r: $rtxt \t nid: $(n.id)")
            disc *= discount(pomdp)
            sumr += r*disc
            s = sprime
            if !hasnext(n)
                sim.display && println("===========")
                break
            end
            n = update(updater, n, n.act, obs)
        end
        sum_reward += sumr
    end
    sum_reward /= sim.times

    if isapprox(sum_reward, 0)
        @warn("Simulated reward close to 0")
    end
    return sum_reward
end

function simulate(sim::MCVISimulator, pomdp::POMDPs.POMDP, policy::POMDPs.Policy, updater::POMDPs.Updater, initial_belief::MCVIBelief)
    sum_reward::Reward = 0
    if sim.init_state != nothing
        s = sim.init_state
    else
        s = initialstate(pomdp, sim.rng)
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
            sprime, obs, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)
            disc *= discount(pomdp)
            sumr += r*disc
            s = sprime
            b = next(b, a, pomdp, sim.rng)
            b = next(b, obs, pomdp)
        end
        sum_reward += sumr
    end
    sum_reward /= sim.times

    if isapprox(sum_reward, 0)
        @warn("Simulated reward close to 0")
    end
    return sum_reward
end
