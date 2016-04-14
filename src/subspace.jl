"""
Subspace of particles of states
"""
type MCVISubspace{S,A}
    particles::Vector{POMDPs.State{S}}
    imm_rewards::Vector{POMDPs.Reward}
    next_state::Dict{POMDPs.Action{A},MCVISubspace{S,A}}
end

"""
Number of particles in the subspace
"""
Base.length(ss::MCVISubspace{S,A}) = length(ss.particles)

"""
    particle(ss::MCVISubspace{S,A}, i::Int64)

Returns i-th particle for the given subspace
"""
particle(ss::MCVISubspace{S,A}, i::Int64) = ss.particles[i]

"""
    weights(ss::MCVISubspace{S,A}, obs::POMDPs.Observation{O}), pomdp::POMDPs.POMDP{S,A,O}

Returns an observation's weight for each particle in the subspace
"""
function weights(ss::MCVISubspace{S,A}, obs::POMDPs.Observation{O}, pomdp::POMDPs.POMDP{S,A,O})
    wts = zeros(length(ss))
    for i in 1:length(wts)
        wts[i] = weight(pomdp, particle(ss, i), obs)
    end
    return wts
end

"""
    create_next(ss::MCVISubspace{S,A}, act::POMDPs.Action{A}), pomdp::POMDPs.POMDP{S,A,O}

Creates the next subspace for an action
"""
function create_next(ss::MCVISubspace{S,A}, act::POMDPs.Action{A}, pomdp::POMDPs.POMDP{S,A,O})
    next_particles = Vector{POMDPs.State{S}}(length(ss))
    imm_rs = zeros(length(ss))

    for (i, s) in enumerate(ss.particles)
        (next_particles[i], imm_rs[i]) = generate_sr(pomdp, s, act, pomdp.rng)
    end
    return MCVISubspace(next_particles, im_rs, Dict{POMDPs.Action{A}, MCVISubspace{S,A}}())
end

"""
    next(ss::MCVISubspace{S,A}, act::POMDPs.Action{A}), pomdp:POMDPs.POMDP{S,A,O}

Returns the next subspace according to the action. If it does not exist, creates one.
"""
function next(ss::MCVISubspace{S,A}, act::POMDPs.Action{A}, pomdp:POMDPs.POMDP{S,A,O})
    local next_ss::MCVISubspace{S,A}
    if haskey(ss.next_state, act)
        next_ss = ss.next_state[act]
    else
        next_ss = create_next(ss, pomdp, act)
        ss.next_state[act] = next_ss
    end
    return next_ss
end

"""
    immediate_reward(ss::MCVISubspace{S,A}, weights::Vector{Float64})

Returns the weighted immediate reward from a subspace
"""
function immediate_reward(ss::MCVISubspace{S,A}, weights::Vector{Float64})
    if isempty(ss.imm_rewards)
        return 0
    end
    return dot(weights, ss.imm_rewards)
end
