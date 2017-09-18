mutable struct AlphaEdge
    vec::Vector{Float64}
    id::UInt64
end
Base.hash(a::AlphaEdge, h::UInt64=zero(UInt64)) = hash(a.vec, hash(a.id, h))

mutable struct MCVINode
    id::UInt64
    act::Any
    states::Any
    alpha_edges::Vector{AlphaEdge}
    MCVINode() = new()
    MCVINode(id, act, states, aes) = new(id, act, states, aes)
end

hasnext(n::MCVINode) = !isempty(n.alpha_edges)

copy(n::MCVINode) = MCVINode(n.id, n.act, n.states, n.alpha_edges)

"""
Returns the next node in the policygraph
"""
mutable struct MCVIUpdater{P<:POMDP} <: POMDPs.Updater
    problem::P
    root
    root_belief
    nodes::Dict{UInt64, MCVINode}
    nodes_queue::Vector{MCVINode}
end

MCVIUpdater(problem) = MCVIUpdater(problem, nothing, nothing, Dict{UInt64, MCVINode}(), Vector{MCVINode}())

Base.length(ps::MCVIUpdater) = length(ps.nodes)

hasroot(ps::MCVIUpdater) = ps.root != nothing

hasnode(ps::MCVIUpdater, n::MCVINode) = haskey(ps.nodes, n.id)

create_belief(ps::MCVIUpdater) = MCVINode()

function initialize_belief(up::MCVIUpdater, b::Any)
    if up.root_belief != b
        if @implemented initial_state_distribution(::typeof(up.problem))
            is = initial_state_distribution(up.problem)
        else
            is = "<initial_state_distribution(::$(typeof(up.problem))) not implemented>"
        end
        warn("""
             The belief used to start MCVI policy execution was (potentially) different from the initial belief used in the MCVI solution.

             updater root belief: $(up.root_belief)
             b0 for policy execution: $b
             initial_state_distribution(pomdp): $is
             """)
    end
    return up.root
end



function create_node{A}(ps::MCVIUpdater, a::A, states::Any, alpha_edges::Vector{AlphaEdge})
    if states == nothing
        st_hash = Base.hash(states)
    else
        st_hash = hash(states) # TODO: recursive hash of array?
    end
    id = hash(a, hash(alpha_edges, st_hash))

    return MCVINode(id, a, states, alpha_edges)
end


mutable struct MCVIPolicy <: POMDPs.Policy
    problem::POMDPs.POMDP
    updater::Union{Void, POMDPs.Updater}
    MCVIPolicy() = new()
    MCVIPolicy(p) = new(p, nothing)
    MCVIPolicy(p, up) = new(p, up)
    MCVIPolicy(p, up, root) = new(p, up)
end


function init_node(ps::MCVIUpdater, problem::POMDPs.POMDP)
    return create_node(ps, init_lower_action(problem), nothing, Vector{AlphaEdge}())
end

function init_nodes(policy::MCVIPolicy, ps::MCVIUpdater)
    ns = Vector{MCVINode}()
    for a in iterator(actions(policy.problem))
        if isterminal(policy.problem, a) # FIXME: Can use init_lower_action instead?
            push!(ns, create_node(ps, a, nothing, Vector{AlphaEdge}()))
        end
    end
    return ns
end

function addnode!(ps::MCVIUpdater, n::MCVINode)
    ps.nodes[n.id] = n
    push!(ps.nodes_queue, n)
end


updater(policy::MCVIPolicy) = policy.updater

"""
Initialize and return the MCVIUpdater
"""
function initialize_updater!(policy::MCVIPolicy)
    ps = MCVIUpdater(policy.problem)
    ns = init_nodes(policy, ps)
    for n in ns
        addnode!(ps, n)
        ps.root = n
    end
    policy.updater = ps
end

"""
Move to the next policy state given observation
"""
function update{A,O}(ps::MCVIUpdater, n::MCVINode, act::A, obs::O, np::MCVINode=create_belief(ps))
    @assert hasnext(n) "No next policy state exists"

    local nid::UInt64
    if length(n.alpha_edges) == 1
        nid = n.alpha_edges[1].id
    else
        @assert n.states != nothing "No states in the current node"
        obs_wt = zeros(length(n.states))
        for (i,s) in enumerate(n.states)
            obs_wt[i] = obs_weight(ps.problem, act, s, obs)
        end
        maxv = -Inf
        for ae in n.alpha_edges
            v = dot(obs_wt, ae.vec)
            if v > maxv
                maxv = v
                nid = ae.id
            end
        end
    end
    @assert ps.nodes[nid] != nothing
    np = ps.nodes[nid]
    return np
end

"""
Return all next policy states for the given node
"""
function next_nodes(ps::MCVIUpdater, n::MCVINode)
    nodes = Vector{MCVINode}(length(n.alpha_edges))
    for i in 1:length(nodes)
        nid = n.alpha_edges[i].id
        nodes[i] = ps.nodes[nid]
    end
    return nodes
end


function to_dict(n::MCVINode)
    return Dict("id"          => n.id,
                "act"         => n.act,
                "states"      => n.states,
                "alpha_edges" => n.alpha_edges)
end

"""
Dump policy graph for inspection
"""
function dump_json(policy::MCVIPolicy, filename::AbstractString)
    root = policy.updater.root
    nodes = policy.updater.nodes
    open(filename, "w") do f
        write(f, JSON.json(Dict(root.id => to_dict(root))))
        for k in keys(nodes)
            write(f, JSON.json(Dict(k => to_dict(nodes[k]))))
        end
    end
end
