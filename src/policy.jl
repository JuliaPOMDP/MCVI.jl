type AlphaEdge
    vec::Vector{Float64}
    id::UInt64
end
Base.hash(a::AlphaEdge, h::UInt64=zero(UInt64)) = hash(a.vec, hash(a.id, h))

type MCVINode
    id::UInt64
    act::Any
    states::Any
    alpha_edges::Vector{AlphaEdge}
end

hasnext(n::MCVINode) = !isempty(n.alpha_edges)

copy(n::MCVINode) = MCVINode(n.id, n.act, n.states, n.alpha_edges)


"""
Returns the next node in the policygraph
"""
type MCVIUpdater{S,A} <: POMDPs.Updater{MCVINode}
    root
    nodes::Dict{UInt64, MCVINode}
    nodes_queue::Vector{MCVINode}
end

MCVIUpdater{S,A}(pomdp::POMDPs.POMDP{S,A}) = MCVIUpdater{S,A}(nothing, Dict{UInt64, MCVINode}(), Vector{MCVINode}())


Base.length(ps::MCVIUpdater) = length(ps.nodes)

hasroot(ps::MCVIUpdater) = ps.root != nothing

hasnode(ps::MCVIUpdater, n::MCVINode) = haskey(ps.nodes, n.id)

function create_node{S,A}(ps::MCVIUpdater{S,A}, a::A, states, alpha_edges::Vector{AlphaEdge})
    if states == nothing
        st_hash = Base.hash(states)
    else
        st_hash = hash(states) # TODO: recursive hash of array?
    end
    id = hash(a, hash(alpha_edges, st_hash))

    return MCVINode(id, a, states, alpha_edges)
end


type MCVIPolicy <: POMDPs.Policy
    problem::POMDPs.POMDP
    updater::Union{Void, POMDPs.Updater}
    MCVIPolicy() = new()
    MCVIPolicy(p) = new(p, nothing)
    MCVIPolicy(p, up) = new(p, up)
    MCVIPolicy(p, up, root) = new(p, up)
end


function init_node{S,A}(ps::MCVIUpdater{S,A}, problem::POMDPs.POMDP)
    return create_node(ps, init_lower_action(problem), nothing, Vector{AlphaEdge}())
end

function init_nodes{S,A}(policy::MCVIPolicy, ps::MCVIUpdater{S,A})
    ns = Vector{MCVINode}()
    for a in iterator(actions(policy.problem))
        if isterminal(policy.problem, a)
            push!(ns, create_node(ps, a, nothing, Vector{AlphaEdge}()))
        end
    end
    return ns
end

function addnode!{S,A}(ps::MCVIUpdater{S,A}, n::MCVINode)
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
function update{A,O}(ps::MCVIUpdater, n::MCVINode, act::A, obs::O)
    @assert hasnext(n) "No next policy state exists"

    local nid::UInt64
    if length(n.alpha_edges) == 1
        nid = n.alpha_edges[1].id
    else
        @assert n.states != nothing "No states in the current node"
        obs_wt = zeros(length(n.states))
        for (i,s) in enumerate(n.states)
            obs_wt[i] = pdf(s, obs) # Observation weight
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
    return ps.nodes[nid]
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
