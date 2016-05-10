module MCVI

using JSON
using POMDPs
import GenerativeModels: generate_sor, generate_o, initial_state

import POMDPs: solve, action, create_policy, rand, simulate, updater

typealias Reward Float64

function lowerbound{S}(p::POMDPs.POMDP, s::S, rng::AbstractRNG)
    error("`lowerbound` Not Implemented")
end

function upperbound{S}(p::POMDPs.POMDP, s::S, rng::AbstractRNG)
    error("`upperbound` Not Implemented")
end

function init_lower_action{S,A}(p::POMDPs.POMDP{S,A})
    error("`init_lower_action` Not Implemented")
end

include("subspace.jl")
include("belief.jl")
include("policy.jl")
include("simulate.jl")
include("alphaedge.jl")
include("updater.jl")
include("solver.jl")


export MCVISolver, MCVIPolicy, solve, action, create_policy, MCVIUpdater, MCVISimulator, updater, dump_json

end # module
