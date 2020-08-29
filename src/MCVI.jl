module MCVI

using JSON
using POMDPs
using ParticleFilters
using Random
using Printf
using Distributed
using LinearAlgebra
import Statistics
import POMDPs: solve, action, rand, simulate, updater, initialize_belief, update

using POMDPLinter: @POMDP_require, @implemented

# Currently the only method of ParticleFilters.obs_weight is obs_weight(m, s, a, sp, o)
# Since MCVI uses obs_weight(m, a, sp, o) and obs_weight(m, sp, o), we define them here
# MCVI should be fixed to use the s, a, sp version.
ParticleFilters.obs_weight(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
ParticleFilters.obs_weight(m::POMDP, sp, o) = pdf(observation(m, sp), o)

const Reward = Float64

# implementation warning handled by POMDPrequire
function init_lower_action end

include("subspace.jl")
include("belief.jl")
include("policy.jl")
include("simulate.jl")
include("alphaedge.jl")
include("updater.jl")
include("solver.jl")


export MCVISolver,
       MCVIPolicy,

       solve,
       action,
       MCVIUpdater,
       MCVISimulator,
       updater,
       dump_json,
       upper_bound,
       lower_bound

end # module
