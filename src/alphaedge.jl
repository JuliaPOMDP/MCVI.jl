"""
Evaluate value of the graph with a given root node
"""
function evaluate(policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP)
    N = 5000                    # TODO: pompd/solver config
    n = policy.updater.root
    v = 0.0
    v = @parallel (+) for i in 1:N
        # @assert sim.init_state == nothing
        simulate(sim, pomdp, policy, policy.updater, n)
    end
    v /= N
    return v
end

"""
Evaluate a batch of states
"""
function evaluate{S}(sts::Vector{S}, policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP, n::MCVINode)
    vs = zeros(length(sts))
    for (i, s) in enumerate(sts)
        sim.init_state = s
        vs[i] += simulate(sim, pomdp, policy, policy.updater, n)
    end
    return vs
end

"""
Evaluate belief
"""
function evaluate(b::MCVIBelief, policy::POMDPs.Policy, sim::MCVISimulator, pomdp::POMDPs.POMDP, n::MCVINode, num_eval_belief::Int64)
    # TODO num_eval_belief, possibly pomdp.config/solver
    # num_eval_belief = 500
    val::Float64 = 0.0
    for i in 1:num_eval_belief
        s = rand(sim.rng, b)    # This is the initial state stuff :/
        sim.init_state = s      # TODO maybe roll this into simulate as well?
        val += simulate(sim, pomdp, policy, policy.updater, n)
    end
    sim.init_state = nothing    # Back to being nothing
    return val/num_eval_belief
end

"""
Alpha vectors
"""
function compute{S}(sts::Vector{S}, policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP, n::MCVINode)
    a = evaluate(sts, updater, sim, pomdp, n)
    edge = AlphaEdge(a, n.id)
    return edge
end

"""
Least square computation
"""
function compute{S,A,O}(sts::Vector{S}, policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP{S,A,O}, n::MCVINode, ba::MCVIBelief)
    # XXX possibly pomdp/solver config
    num_states = 1000
    num_obs = 50

    obs = Vector{O}(num_obs)
    ov = zeros(num_obs)
    osum = zeros(num_obs)

    for i in 1:num_obs
        # sample observation from belief
        s = rand(pomdp.rng, ba) # TODO: which rng?
        obs[i] = generate_o(pomdp, nothing, nothing, s, pomdp.rng)
    end

    for i in 1:num_states
        s =rand(sim.rng, ba)
        sim.init_state = s
        v = simulate(sim, pomdp, policy, policy.updater, n)
        for (j,o) in enumerate(obs)
            wt = pdf(s, o)
            osum[j] += wt
            ov[j] += v*wt
        end
    end
    # normalize
    for (i,p) in enumerate(osum)
        ov[i] /= p
    end

    X = zeros(num_obs, length(sts))
    for (i,o) in enumerate(obs)
        for (j,s) in enumerate(sts)
            X[i,j] = pdf(s, o)
        end
    end

    b = X \ ov                  # Least squares

    @assert length(b) == length(sts)
    @assert !isnan(b[1])
    # @assert isapprox(X*b, ov) "Error in least squares: \n$(ov) \n$(X*b)" 

    edge = AlphaEdge(b, n.id)
    return edge
end
