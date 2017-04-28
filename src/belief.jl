"""
Particle belief over subspaces
"""
type MCVIBelief{S,A}
    space::MCVISubspace{S,A}
    weights::Vector{Float64}
    id::UInt64
end

MCVIBelief{S,A}(space::MCVISubspace{S,A}...) = MCVIBelief{S,A}(space...)

Base.length(b::MCVIBelief) = length(b.weights)

"""
Create initial belief particles
Is supposed to represent the initial state distribution.
Not sure how to propagate num_particles without storing it in pomdp or calling it a different function?
"""
function initial_belief{S,A,O}(pomdp::POMDPs.POMDP{S,A,O}, num_particles::Int64, rng::AbstractRNG)
    particles = Vector{S}(num_particles)
    wts = zeros(num_particles)
    for i in 1:num_particles
        particles[i] = initial_state(pomdp, rng) # Sample init state
        wts[i]= 1.0/float(num_particles)
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
function next{S,A}(bb::MCVIBelief{S,A}, act::A, pomdp::POMDPs.POMDP, rng::AbstractRNG)
    next_ss = next(bb.space, act, pomdp, rng) # Get the next subspace
    return MCVIBelief(next_ss, bb.weights, UInt64(0)) # FIXME update id?
end

"""
    next(bb::MCVIBelief{S,A}, obs::POMDPs.Observation{O}, pomdp::POMDPs.POMDP{S,A,O})

Get the next belief according to observation.
Observations update the weights of the space created by actions.
"""
function next{S,O}(bb::MCVIBelief{S}, obs::O, pomdp::POMDPs.POMDP)
    wts = weights(bb.space, obs, pomdp) # Get weights for the observation
    belief_after_obs = bb.weights .* wts
    obs_norm = sum(belief_after_obs)
    @assert obs_norm != 0 "Normalizing constant should not be zero"
    belief_after_obs /= obs_norm
    return MCVIBelief(bb.space, belief_after_obs, UInt64(0))
end


# """
#     lower_bound(b::MCVIBelief{S}, pomdp::POMDPs.POMDP, rng::AbstractRNG)
# 
# Provides the minimum sum of particle values in a belief
# """
function lower_bound(lb::Any, pomdp::POMDPs.POMDP, b::MCVIBelief)
    sum = 0.0
    for (i, w) in enumerate(b.weights)
        s = particle(b.space, i)
        w = b.weights[i]
        sum += lower_bound(lb, pomdp, s) * w # Weighted lower bound for the state
    end
    return sum
end

# """
#     upper_bound(b::MCVIBelief{S}, pomdp::POMDPs.POMDP, rng::AbstractRNG)
# 
# Provides the maximum sum of particle values in a belief
# """
function upper_bound(ub::Any, pomdp::POMDPs.POMDP, b::MCVIBelief)
    sum = 0.0
    for (i, w) in enumerate(b.weights)
        s = particle(b.space, i)
        w = b.weights[i]
        sum += upper_bound(ub, pomdp, s) * w # Weighted upper bound for the state
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
