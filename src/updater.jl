type BeliefActionBackup{S,A}
    act::A
    ba::MCVIBelief
    sa::Vector{S}

    alpha_edges::Vector{AlphaEdge}
    to_append
    n                           # MCVINode
end


type BeliefBeliefBackup
    belief::MCVIBelief
    act_backupers::Vector{BeliefActionBackup}
    last_index::Int64
    max_node
    maxv::Float64
end

function BeliefBeliefBackup(belief, pomdp::POMDPs.POMDP)
    num_state = 5000            # TODO get num_state from pomdp
    bb = BeliefBeliefBackup(belief, Vector{BeliefActionBackup}(), 1, nothing, -Inf)
    for a in iterator(actions(pomdp))
        if a == init_lower_action(pomdp)
            continue
        end
        pb = next(belief, a, pomdp)
        sa = [rand(pomdp.rng, pb) for i in 1:num_state] # TODO get num_state from pomdp
        ac = BeliefActionBackup(a, pb, sa, Vector{AlphaEdge}(), nothing, nothing)
        push!(bb.act_backupers, ac)
    end
    return bb
end

function prune_alpha_edges(alpha_edges, actback, pomdp)
    ba = actback.ba
    sa = actback.sa
    keep = Vector{Bool}(length(alpha_edges))
    #TODO: num_prune_obs : pomdp/solver config
    num_prune_obs = 1000
    for i in 1:num_prune_obs
        # Sample observation from belief
        obs = generate_o(pomdp, nothing, nothing, rand(pomdp.rng, ba), pomdp.rng) # TODO: rng?
        obswt = zeros(length(sa))
        for (k,s) in enumerate(sa)
            obswt[k] = pdf(s, obs)
        end
        maxv = -Inf
        local maxj::Int64
        for (j, a) in enumerate(alpha_edges)
            v = dot(obswt, a.vec)
            if v > maxv
                maxv = v
                maxj = j
            end
        end
        keep[maxj]= true
    end
    pruned = Vector{AlphaEdge}()
    for (i, ae) in enumerate(alpha_edges)
        if keep[i]
            push!(pruned, ae)
        end
    end
    return pruned
end

"""
Calls the least square compute function
"""
function compute_alpha_edges(nodes, actback, policy, sim, pomdp::POMDPs.POMDP)
    ba = actback.ba
    sa = actback.sa
    alpha_edges = Vector{AlphaEdge}(length(nodes))

    for i in 1:length(nodes)
        n = nodes[i]
        alpha_edges[i] = compute(sa, policy, sim, pomdp, n, ba) # FIXME Slowwww
    end
    return alpha_edges
end

function add_alpha_edges!(edges, actback, updater, pomdp::POMDPs.POMDP)
    # FIXME side effects on actback?
    new_alpha_edges = deepcopy(actback.alpha_edges)
    append!(new_alpha_edges, edges)
    # Prune
    new_alpha_edges = prune_alpha_edges(new_alpha_edges, actback, pomdp) # Slow?

    # check
    l1 = length(actback.alpha_edges)
    l2 = length(new_alpha_edges)
    if (l1 == l2) && actback.alpha_edges[l1].id == new_alpha_edges[l2].id
        return actback.n
    end

    actback.alpha_edges = new_alpha_edges
    actback.n = create_node(updater, actback.act, actback.sa, actback.alpha_edges)
    return actback.n
end

"""
Backup action
"""
function backup(actback, policy, sim, pomdp, nodes)
    # Compute alpha edges
    new_alpha_edges = compute_alpha_edges(nodes, actback, policy, sim, pomdp)
    # Add alpha edges
    n = add_alpha_edges!(new_alpha_edges, actback, policy.updater, pomdp)
    return n
end

"""
Returns the best node and its value
"""
function find_best_node(nodes, policy, sim, pomdp::POMDPs.POMDP, belief)
    @assert length(nodes) > 0
    vs = zeros(length(nodes))
    vs = pmap((n)->evaluate(belief, policy, sim, pomdp, n), nodes)
    # for i in 1:length(nodes)
    #     n = nodes[i]
    #     vs[i] = evaluate(belief, policy, sim, pomdp, n)
    # end
    (maxv, maxi) = findmax(vs)
    max_node = nodes[maxi]
    return (max_node, maxv)
end

"""
Update belief backup's best node
"""
function update!(bb::BeliefBeliefBackup, nodes::Vector{MCVINode}, pomdp::POMDPs.POMDP, policy::MCVIPolicy, sim::MCVISimulator, eps::Float64)
    (n, v) = find_best_node(nodes, policy, sim, pomdp, bb.belief)
    if bb.maxv + eps < v
        bb.maxv = v
        bb.max_node = n
    end
end

"""
Backup belief
"""
function backup(bb::BeliefBeliefBackup, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP)
    # bb = BeliefBeliefBackup(belief, pomdp)
    # Get newer nodes
    nodes = policy.updater.nodes_queue[bb.last_index:end]
    # Update best belief
    update!(bb, nodes, pomdp, policy, sim, -0.1) # Try \epsilon less
    print_with_color(:cyan, "update")
    println(" (nodes)")

    # Get new nodes from action backup
    new_nodes = Vector{MCVINode}(length(bb.act_backupers))
    for (i, actback) in enumerate(bb.act_backupers)
        new_nodes[i] = backup(actback, policy, sim, pomdp, nodes) # Backup action, TODO Slowwww
    end
    print_with_color(:cyan, "backup action")
    println(" (nodes)")

    bb.last_index += length(nodes)
    update!(bb, new_nodes, pomdp, policy, sim, +0.1) # Try \epsilon more

    return (bb.max_node, bb.maxv)                     # TODO return bb ?
end

function backup(belief::MCVIBelief, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP)
    # Belief backup struct
    bb = BeliefBeliefBackup(belief, pomdp)
    return backup(bb, policy, sim, pomdp)
end
