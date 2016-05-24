type MCVIActionBackup{S,A}
    act::A
    ba::MCVIBelief
    sa::Vector{S}

    alpha_edges::Vector{AlphaEdge}
    to_append
    n                           # MCVINode
end


type MCVIBeliefBackup
    belief::MCVIBelief
    act_backupers::Vector{MCVIActionBackup}
    last_index::Int64
    max_node
    maxv::Float64
end

MCVIBeliefBackup(belief::MCVIBelief) = MCVIBeliefBackup(belief, Vector{MCVIActionBackup}(), 1, nothing, -Inf)

function initialize_belief_backup!{S}(bb::MCVIBeliefBackup, pomdp::POMDPs.POMDP{S}, num_state::Int64, rng::AbstractRNG)
    for a in iterator(actions(pomdp))
        if a == init_lower_action(pomdp)
            continue
        end
        pb = next(bb.belief, a, pomdp, rng)
        sa = S[rand(rng, pb) for i in 1:num_state]
        ac = MCVIActionBackup(a, pb, sa, Vector{AlphaEdge}(), nothing, nothing)
        push!(bb.act_backupers, ac)
    end
    return bb
end

function prune_alpha_edges{S,A}(alpha_edges, actback::MCVIActionBackup{S,A}, pomdp::POMDPs.POMDP, num_prune_obs::Int64, rng::AbstractRNG)
    ba = actback.ba
    sa = actback.sa
    keep = Vector{Bool}(length(alpha_edges))
    for i in 1:num_prune_obs
        # Sample observation from belief
        obs = generate_o(pomdp, nothing, nothing, rand(rng, ba), rng)
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
function compute_alpha_edges{S,A}(nodes::Vector{MCVINode}, actback::MCVIActionBackup{S,A}, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP, scratch::Scratch)
    ba = actback.ba
    sa = actback.sa
    scratch.X = zeros(size(scratch.X,1), length(sa))
    alpha_edges = Vector{AlphaEdge}(length(nodes))
    # println("len_nodes: $(length(nodes))")
    tic()
    alpha_edges = pmap((n)->compute(sa, policy, sim, pomdp, n), nodes) # FIXME simple
    # for (i,n) in enumerate(nodes)
    #     alpha_edges[i] = compute(sa, policy, sim, pomdp, n)
    # end

    # alpha_edges = pmap((n)->compute(sa, policy, sim, pomdp, n, ba, scratch), nodes)
    # for i in 1:length(nodes)
    #     n = nodes[i]
    #     alpha_edges[i] = compute(sa, policy, sim, pomdp, n, ba, scratch) # FIXME Slowwww
    # end
    toc()
    return alpha_edges
end

function add_alpha_edges!{S,A}(actback::MCVIActionBackup{S,A}, edges, updater::MCVIUpdater, pomdp::POMDPs.POMDP, num_prune_obs::Int64, rng::AbstractRNG)
    new_alpha_edges = deepcopy(actback.alpha_edges)
    append!(new_alpha_edges, edges)
    # Prune
    new_alpha_edges = prune_alpha_edges(new_alpha_edges, actback, pomdp, num_prune_obs, rng) # Slow?

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
function backup{S,A}(actback::MCVIActionBackup{S,A}, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP, nodes, num_prune_obs::Int64, scratch::Scratch)
    # Compute alpha edges
    new_alpha_edges = compute_alpha_edges(nodes, actback, policy, sim, pomdp, scratch)
    # Add alpha edges
    n = add_alpha_edges!(actback, new_alpha_edges, policy.updater, pomdp, num_prune_obs, sim.rng)
    return n
end

"""
Returns the best node and its value
"""
function find_best_node(nodes::Vector{MCVINode}, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP, belief::MCVIBelief, num_eval_belief::Int64)
    @assert length(nodes) > 0
    vs = pmap((n)->evaluate(belief, policy, sim, pomdp, n, num_eval_belief), nodes)
    # vs = zeros(length(nodes))
    # for i in 1:length(nodes)
    #     n = nodes[i]
    #     vs[i] = evaluate(belief, policy, sim, pomdp, n, num_eval_belief)
    # end
    (maxv, maxi) = findmax(vs)
    max_node = nodes[maxi]
    return (max_node, maxv)
end

"""
Update belief backup's best node
"""
function update!(bb::MCVIBeliefBackup, nodes::Vector{MCVINode}, pomdp::POMDPs.POMDP, policy::MCVIPolicy, sim::MCVISimulator, eps::Float64, num_eval_belief::Int64)
    (n, v) = find_best_node(nodes, policy, sim, pomdp, bb.belief, num_eval_belief)
    if bb.maxv + eps < v
        bb.maxv = v
        bb.max_node = n
    end
end

"""
Backup belief
"""
function backup(bb::MCVIBeliefBackup, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP, num_prune_obs::Int64, num_eval_belief::Int64, scratch::Scratch)
    # Get newer nodes
    nodes = policy.updater.nodes_queue[bb.last_index:end]
    # Update best belief
    tic()
    update!(bb, nodes, pomdp, policy, sim, -0.1, num_eval_belief) # Try ϵ less
    print_with_color(:cyan, "update")
    println(" (nodes): $(toq())s")

    # Get new nodes from action backup
    tic()
    new_nodes = Vector{MCVINode}(length(bb.act_backupers))
    for (i, actback) in enumerate(bb.act_backupers)
        new_nodes[i] = backup(actback, policy, sim, pomdp, nodes, num_prune_obs, scratch) # Backup action, FIXME Slowwww
    end
    print_with_color(:cyan, "backup action")
    println(" (nodes): $(toq())s")

    bb.last_index += length(nodes)
    update!(bb, new_nodes, pomdp, policy, sim, +0.1, num_eval_belief) # Try ϵ more

    return (bb.max_node, bb.maxv)
end

function backup(belief::MCVIBelief, policy::MCVIPolicy, sim::MCVISimulator, pomdp::POMDPs.POMDP, num_state::Int64, num_prune_obs::Int64, num_eval_belief::Int64, scratch::Scratch)
    # Belief backup struct
    bb = MCVIBeliefBackup(belief)
    initialize_belief_backup!(bb, pomdp, num_state, sim.rng)
    return backup(bb, policy, sim, pomdp, num_prune_obs, num_eval_belief, scratch)
end
