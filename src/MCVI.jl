module MCVI

using JSON
using POMDPs
using ParticleFilters

import POMDPs: solve, action, rand, simulate, updater, initialize_belief, update

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
