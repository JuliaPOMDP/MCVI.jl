"""
Evaluate value of the graph with a given root node
"""
function evaluate(policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP)
    N = 5000                    # TODO: pompd/solver config
    n = policy.updater.root
    v = 0.0
    v = @parallel (+) for i in 1:N
        simulate(sim, pomdp, policy, policy.updater, n)
    end
    v /= N
    return v
end

"""
Evaluate a batch of states
"""
function evaluate{S}(sts::Vector{S}, policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP, n::MCVINode)
    # vs = pmap((s)->simulate(sim, pomdp, policy, policy.updater, n, s), sts)
    vs = zeros(Float64, length(sts))
    for (i,s) in enumerate(sts)
        vs[i] = simulate(sim, pomdp, policy, policy.updater, n, s)
    end
    return vs
end

"""
Evaluate belief
"""
function evaluate(b::MCVIBelief, policy::POMDPs.Policy, sim::MCVISimulator, pomdp::POMDPs.POMDP, n::MCVINode, num_eval_belief::Int64)
    val::Float64 = 0.0
    val = @parallel (+) for i in 1:num_eval_belief
        s = rand(sim.rng, b)    # This is the initial state stuff :/
        # sim.init_state = s      # TODO maybe roll this into simulate as well?
        simulate(sim, pomdp, policy, policy.updater, n, s)
    end
    # sim.init_state = nothing    # Back to being nothing
    return val/num_eval_belief
end

"""
Alpha vectors
"""
function compute{S,A,O}(sts::Vector{S}, policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP{S,A,O}, n::MCVINode)
    a = evaluate(sts, policy, sim, pomdp, n)
    edge = AlphaEdge(a, n.id)
    return edge
end

function _fill_obs!{S,A,O}(obs::Vector{O}, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP{S,A,O}, ba::MCVIBelief)
    for i in 1:length(obs)
        # sample observation from belief
        s = rand(sim.rng, ba)
        obs[i] = generate_o(pomdp, nothing, nothing, s, sim.rng)
    end
end

function _fill_ov!{O}(ov::Vector{Float64}, osum::Vector{Float64}, obs::Vector{O},
                      policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP,
                      n::MCVINode, ba::MCVIBelief, num_states::Int64)
    sts = [rand(sim.rng, ba) for _ in 1:num_states]
    v = evaluate(sts, policy, sim, pomdp, n)
    for i in 1:num_states
        for (j,o) in enumerate(obs)
            wt = pdf(sts[i], o)
            osum[j] += wt
            ov[j] += v[i]*wt
        end
    end
    # normalize
    for (i,p) in enumerate(osum)
        ov[i] /= p
    end
end

function _fill_X!{O,S}(X::Array{Float64,2}, obs::Vector{O}, sts::Vector{S})
    for (j,s) in enumerate(sts)
        for (i,o) in enumerate(obs)
            X[i,j] = pdf(s, o)
        end
    end
end

type Scratch{O}
    obs::Vector{O}
    ov::Vector{Float64}
    osum::Vector{Float64}
    X::Array{Float64,2}
end

"""
Least square computation
"""
function compute{S,A,O}(sts::Vector{S}, policy::POMDPs.Policy, sim::POMDPs.Simulator, pomdp::POMDPs.POMDP{S,A,O}, n::MCVINode, ba::MCVIBelief, scratch::Scratch)
    _fill_obs!(scratch.obs, sim, pomdp, ba)
    _fill_ov!(scratch.ov, scratch.osum, scratch.obs, policy, sim, pomdp, n, ba, size(scratch.X,2))
    _fill_X!(scratch.X, scratch.obs, sts)

    b = scratch.X \ scratch.ov                  # Least squares

    @assert length(b) == length(sts)
    @assert !isnan(b[1])

    edge = AlphaEdge(b, n.id)
    return edge
end
