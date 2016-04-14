function action(policy::MCVIPolicy, node::MCVINode)
    return node.act             # Confused now. What should be the second argument again?
end

abstract TreeNode

type BeliefNode <: TreeNode
    obs
    belief:: MCVIBelief
    upper::Reward
    lower::Reward
    best_node                   # MCVINode
    children::Vector{TreeNode}
end

type ActionNode{A} <: TreeNode
    act::A
    belief:: MCVIBelief
    upper::Reward
    imm_reward::Reward
    children::Vector{BeliefNode}
end

"""

Hyperparameters:
    - `n_iter`          : Number of iterations
    - `num_belief`      : Number of belief particles to be used
    - `obs_branch`      : Branching factor [default 8?]
    - `num_state`       : Number of states to sample from belief [default 500?]
    - `num_prune_obs`   : Number of times to sample observation while pruning alpha edges [default 1000?]
    - `num_eval_belief` : Number of times to simulate while evaluating belief [default 5000?]
"""
type MCVISolver <: POMDPs.Solver
    # updater
    simulator
    root::BeliefNode
    rng::AbstractRNG

    n_iter::Int64
    num_belief::Int64
    obs_branch::Int64
    num_state::Int64
    num_prune_obs::Int64
    num_eval_belief::Int64
end

function MCVISolver(pomdp::POMDPs.POMDP, simulator::MCVISimulator, rng::AbstractRNG, n_iter::Int64, num_belief::Int64, obs_branch::Int64, num_state::Int64, num_prune_obs::Int64, num_eval_belief::Int64)
    b0 = initial_belief(pomdp)  # TODO Send num_belief
    upper = upperbound(b0, pomdp)
    lower = lowerbound(b0, pomdp)
    root = BeliefNode(nothing, b0, upper, lower, nothing, Vector{TreeNode}())
    return MCVISolver(simulator, root, rng, n_iter, num_belief, obs_branch, num_state, num_prune_obs, num_eval_belief)
end

create_policy(::MCVISolver, p::POMDPs.POMDP) = MCVIPolicy(p)

"""
Expand beliefs (Add new action nodes)
"""
function expand!(bn::BeliefNode, solver, pomdp)
    if !isempty(bn.children)
        return nothing
    end

    for a in iterator(actions(pomdp))
        bel = next(bn.belief, a, pomdp) # Next belief by action
        imm_r = reward(bel, pomdp)
        local upper::Float64
        if isterminal(pomdp, a)
            upper = imm_r*discount(pomdp)
        else
            # Initialize using problem upper value
            upper = upperbound(bel, pomdp)
        end
        print_with_color(:yellow, "expand")
        println(" (belief) -> $(a) \t $(imm_r) \t $(upper)")
        act_node = ActionNode(a, bel, upper, imm_r, Vector{BeliefNode}())
        push!(bn.children, act_node)
    end
end

function expand!(an::ActionNode, solver, pomdp)
    if !isempty(an.children)
        return nothing
    end
    for i in 1:solver.obs_branch # branching factor
        # Sample observation
        s = rand(pomdp.rng, an.belief)
        obs = generate_o(pomdp, nothing, nothing, s, pomdp.rng) # TODO: rng?
        bel = next(an.belief, obs, pomdp) # Next belief by observation

        upper = upperbound(bel, pomdp)
        lower = lowerbound(bel, pomdp)

        belief_node = BeliefNode(obs, bel, upper, lower, nothing, Vector{ActionNode}())
        push!(an.children, belief_node)
    end
end

"""
Backup over belief
"""
function backup!(bn::BeliefNode, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP)
    # Upper value
    u = -Inf
    for a in bn.children
        if u < a.upper
            u = a.upper
        end
    end
    if bn.upper > u
        bn.upper = u
    end

    # Increase lower value
    policy_node, node_val = backup(bn.belief, policy, solver.simulator, pomdp) # Backup belief
    print_with_color(:magenta, "backup")
    println(" (belief) -> $(node_val) \t $(bn.lower)")
    if node_val > bn.lower
        bn.lower = node_val
        bn.best_node = policy_node
        addnode!(policy.updater, policy_node) # Add node to policy graph
    end
end

"""
Backup over action
"""
function backup!(an::ActionNode, solver::MCVISolver, pomdp::POMDPs.POMDP)
    u::Float64 = 0.0
    for b in an.children
        u += b.upper
    end
    u /= length(an.children)
    u = (u + an.imm_reward) * discount(pomdp)
    if an.upper > u
        an.upper = u
    end
end

"""
Search over belief
"""
function search!(bn::BeliefNode, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP, target_gap::Float64)
    println("belief -> $(bn.obs) \t $(bn.upper) \t $(bn.lower)")
    if (bn.upper - bn.lower) > target_gap
        # Add child action nodes to belief node
        expand!(bn, solver, pomdp)
        max_upper = -Inf
        local choice::ActionNode
        for ac in bn.children
            # Backup action
            backup!(ac, solver, pomdp)
            # Choose the one with max upper limit
            if max_upper < ac.upper
                max_upper = ac.upper
                choice = ac
            end
        end
        # Seach over action
        search!(choice, solver, policy, pomdp, target_gap)
    end
    # backup belief
    backup!(bn, solver, policy, pomdp)
end

"""
Search over action
"""
function search!(an::ActionNode, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP, target_gap::Float64)
    println("act -> $(an.act) \t $(an.upper)")
    if isterminal(pomdp, an.act)
        return nothing
    end
    # Expand action
    expand!(an, solver, pomdp)
    max_gap = 0.0
    local choice = nothing
    for b in an.children
        gap = b.upper - b.lower
        # Choose the belief that maximizes the gap bw upper and lower
        if gap > max_gap
            max_gap = gap
            choice = b
        end
    end
    # If we found anything that improved the difference
    if choice != nothing
        search!(choice, solver, policy, pomdp, target_gap/discount(pomdp))
    else
        println("Gap closed!")
    end
    # Backup action
    backup!(an, solver, pomdp)
end

function solve(solver::MCVISolver, pomdp::POMDPs.POMDP, policy::MCVIPolicy=create_policy(solver, pomdp))
    # Gap between upper and lower
    target_gap = 0.0
    if policy.updater == nothing
        initialize_updater!(policy)
    end
    # Search
    for i in 1:solver.n_iter
        search!(solver.root, solver, policy, pomdp, target_gap) # Here solver.root is a BeliefNode
        policy.root = solver.root.best_node             # Here policy.root is a MCVINode
        dump_json(policy, "/tmp/policy.json")
        if (solver.root.upper - solver.root.lower) < 0.1
            break
        end
        print_with_color(:green, "iter $(i) \t")
        println("upper: $(solver.root.upper) \t lower: $(solver.root.lower)")
    end
    return policy
end
