"""
Particle belief over subspaces
"""
type MCVIBelief{S,A}
    space::MCVISubspace{S,A}
    weights::Vector{Float64}
    id::UInt64
end

MCVIBelief{S,A}(space::MCVISubspace{S,A}, weights::Vector{Float64}, id::UInt64) = MCVIBelief{S,A}(space, weights, id)

Base.length(b::MCVIBelief) = length(b.weights)

"""
Create initial belief particles
Is supposed to represent the initial state distribution.
TODO: Should this be part of the problem definition?
      Not sure how to propagate num_particles without storing it in pomdp or calling it a different function?
"""
function initial_belief{S,A}(pomdp::POMDPs.POMDP{S,A})
    num_particles = 500         # TODO: not sure about this
    particles = Vector{S}(num_particles)
    wts = zeros(num_particles)
    for i in 1:num_particles
        particles[i] = initial_state(pomdp, pomdp.rng) # Sample init state
        wts[i]= 1.0/num_particles
    end
    ss = MCVISubspace{S,A}(particles, Vector{Reward}(), Dict{A, MCVISubspace{S,A}}())
    return MCVIBelief(ss, wts, UInt64(0))
end

"""
    rand(rng::AbstractRNG, b::MCVIBelief{S,A})

Samples states from belief
"""
function rand(rng::AbstractRNG, b::MCVIBelief)
    r = Base.rand(rng)
    for (i, p) in enumerate(b.weights)
        r -= p
        if r < 0
            return particle(b.space, i)
        end
    end
    @assert false "Shouldn't reach here"
end

"""
    next(bb::MCVIBelief{S,A}, act::POMDPs.Action{A}, pomdp::POMDPs.POMDP{S,A,O})

Get the next belief according to action.
Actions create a new subspace
"""
function next{S,A}(bb::MCVIBelief{S,A}, act::A, pomdp::POMDPs.POMDP)
    next_ss = next(bb.space, act, pomdp) # Get the next subspace
    return MCVIBelief(next_ss, bb.weights, UInt64(0)) # FIXME update id?
end

"""
    next(bb::MCVIBelief{S,A}, obs::POMDPs.Observation{O}, pomdp::POMDPs.POMDP{S,A,O})

Get the next belief according to observation.
Observations update the weights of the space created by actions.
"""
function next{S,A,O}(bb::MCVIBelief{S,A}, obs::O, pomdp::POMDPs.POMDP)
    wts = weights(bb.space, obs, pomdp) # Get weights for the observation
    belief_after_obs = bb.weights .* wts
    obs_norm = sum(belief_after_obs)
    @assert obs_norm != 0 "Normalizing constant should not be zero"
    belief_after_obs /= obs_norm
    return MCVIBelief(bb.space, belief_after_obs, UInt64(0))
end


"""
    lowerbound(b::MCVIBelief{S}, pomdp::POMDPs.POMDP)

Provides the minimum sum of particle values in a belief
"""
function lowerbound(b::MCVIBelief, pomdp::POMDPs.POMDP)
    sum = 0.0
    for (i, w) in enumerate(b.weights)
        s = particle(b.space, i)
        w = b.weights[i]
        sum += lowerbound(pomdp, s) * w # Weighted lower bound for the state
    end
    return sum
end

"""
    upperbound(b::MCVIBelief{S}, pomdp::POMDPs.POMDP)

Provides the maximum sum of particle values in a belief
"""
function upperbound(b::MCVIBelief, pomdp::POMDPs.POMDP)
    sum = 0.0
    for (i, w) in enumerate(b.weights)
        s = particle(b.space, i)
        w = b.weights[i]
        sum += upperbound(pomdp, s) * w # Weighted upper bound for the state
    end
    return sum
end

"""
    reward(b::MCVIBelief, pomdp::POMDPs.POMDP)

Returns the immediate reward for a given belief
"""
function reward(b::MCVIBelief, pomdp::POMDPs.POMDP)
    return reward(b.space, b.weights)
end

"""
Returns the mean value of the particle belief
"""
function mean(b::MCVIBelief)
    μ = 0.0
    for i in 1:length(b.weights)
        μ += b.weights[i] * particle(b.space, i)
    end
    return μ
end

"""
Returns the standard deviation of the particle belief
"""
function std(b::MCVIBelief, μ::Float64=mean(b))
    σ = 0.0
    for i in 1:length(b.weights)
        r = particle(b.space, i) - μ
        σ += b.weights[i] * (r * r)
    end
    return σ
end
