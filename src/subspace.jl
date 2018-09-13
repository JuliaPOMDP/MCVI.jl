"""
Subspace of particles of states
"""
mutable struct MCVISubspace{S,A}
    particles::Vector{S}
    imm_rewards::Vector{Reward}
    next_state::Dict{A,MCVISubspace{S,A}}
end
"""
MCVISubspace constructor
"""
MCVISubspace(p::Vector{S}, r, ns::Dict{A,MCVISubspace{S,A}}) where {S,A} = MCVISubspace(p, r, ns)


"""
Number of particles in the subspace
"""
Base.length(ss::MCVISubspace) = length(ss.particles)

"""
    particle(ss::MCVISubspace, i::Int64)

Returns i-th particle for the given subspace
"""
particle(ss::MCVISubspace, i::Int64) = ss.particles[i]

"""
    weights(ss::MCVISubspace{S,A}, obs::O), pomdp::POMDPs.POMDP

Returns an observation's weight for each particle in the subspace
"""
function weights(ss::MCVISubspace{S,A}, obs::O, pomdp::POMDPs.POMDP) where {S,A,O}
    wts = zeros(length(ss))
    for i in 1:length(wts)
        wts[i] = obs_weight(pomdp, particle(ss, i), obs)
    end
    return wts
end

"""
    create_next(ss::MCVISubspace{S,A}, act::A, pomdp::POMDPs.POMDP{S,A,O})

Creates the next subspace for an action
"""
function create_next(ss::MCVISubspace{S,A}, act::A, pomdp::POMDPs.POMDP, rng::AbstractRNG) where {S,A}
    next_particles = Vector{S}(length(ss))
    imm_rs = zeros(length(ss))

    for (i, s) in enumerate(ss.particles)
        (next_particles[i], _, imm_rs[i]) = generate_sor(pomdp, s, act, rng)
    end
    return MCVISubspace(next_particles, imm_rs, Dict{A, MCVISubspace{S,A}}())
end

"""
    next(ss::MCVISubspace{S,A}, act::A, pomdp::POMDPs.POMDP{S,A,O})

Returns the next subspace according to the action. If it does not exist, creates one.
"""
function next(ss::MCVISubspace{S,A}, act::A, pomdp::POMDPs.POMDP, rng::AbstractRNG) where {S,A}
    local next_ss::MCVISubspace{S,A}
    if haskey(ss.next_state, act)
        next_ss = ss.next_state[act]
    else
        next_ss = create_next(ss, act, pomdp, rng)
        ss.next_state[act] = next_ss
    end
    return next_ss
end

"""
    reward(ss::MCVISubspace{S,A}, weights::Vector{Float64})

Returns the weighted immediate reward from a subspace
"""
function reward(ss::MCVISubspace, weights::Vector{Float64})
    if isempty(ss.imm_rewards)
        return 0
    end
    return dot(weights, ss.imm_rewards)
end
