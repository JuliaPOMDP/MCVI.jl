function action(policy::MCVIPolicy, node::MCVINode)
    return node.act             # Confused now. What should be the second argument again?
end

abstract TreeNode

type BeliefNode{O} <: TreeNode
    obs::Nullable{O}
    belief:: MCVIBelief
    upper::Reward
    lower::Reward
    best_node::Nullable{MCVINode}
    children::Vector{TreeNode}
end

type ActionNode{A} <: TreeNode
    act::A
    belief:: MCVIBelief
    upper::Reward
    imm_reward::Reward
    children::Vector{BeliefNode}
end

BeliefNode{O}(obs::Nullable{O}, b::MCVIBelief, u::Reward, l::Reward, bn::Nullable{MCVINode}, c::Vector{TreeNode}) = BeliefNode{O}(obs, b, u, l, bn, c)
BeliefNode{O,A}(obs::Nullable{O}, b::MCVIBelief, u::Reward, l::Reward, bn::Nullable{MCVINode}, c::Vector{ActionNode{A}}) = BeliefNode{O}(obs, b, u, l, bn, c)

"""

Hyperparameters:
    - `n_iter`          : Number of iterations
    - `num_particles`   : Number of belief particles to be used
    - `obs_branch`      : Branching factor [default 8?]
    - `num_state`       : Number of states to sample from belief [default 500?]
    - `num_prune_obs`   : Number of times to sample observation while pruning alpha edges [default 1000?]
    - `num_eval_belief` : Number of times to simulate while evaluating belief [default 5000?]
    - `num_obs`         : [default 50?]
"""
type MCVISolver <: POMDPs.Solver
    simulator::POMDPs.Simulator
    root::Nullable{BeliefNode}

    n_iter::Int64
    num_particles::Int64
    obs_branch::Int64
    num_state::Int64
    num_prune_obs::Int64
    num_eval_belief::Int64

    num_obs::Int64
    scratch::Nullable{Scratch}
    MCVISolver() = new()
    function MCVISolver(sim, root, n_iter, nbp, ob, ns, npo, neb, num_obs)
        new(sim, root, n_iter, nbp, ob, ns, npo, neb, num_obs, Nullable{Scratch}())
    end
end

function initialize_root!{S,A,O}(solver::MCVISolver, pomdp::POMDPs.POMDP{S,A,O})
    b0 = initial_belief(pomdp, solver.num_particles, solver.simulator.rng)
    solver.root = BeliefNode(Nullable{O}(), b0, upperbound(b0, pomdp, solver.simulator.rng), lowerbound(b0, pomdp, solver.simulator.rng), Nullable{MCVINode}(), Vector{TreeNode}())
    solver.scratch = Scratch(Vector{O}(solver.num_obs), zeros(solver.num_obs), zeros(solver.num_obs), zeros(solver.num_obs, 2))
end

create_policy(::MCVISolver, p::POMDPs.POMDP) = MCVIPolicy(p)

"""
Expand beliefs (Add new action nodes)
"""
function expand!(bn::BeliefNode, solver::MCVISolver, pomdp::POMDPs.POMDP)
    if !isempty(bn.children)
        return nothing
    end

    for a in iterator(actions(pomdp))
        bel = next(bn.belief, a, pomdp, solver.simulator.rng) # Next belief by action
        imm_r = reward(bel, pomdp)
        local upper::Float64
        if isterminal(pomdp, a) # FIXME This is  necessary?
            upper = imm_r*discount(pomdp)
        else
        # Initialize using problem upper value
            upper = upperbound(bel, pomdp, solver.simulator.rng)
        end
        print_with_color(:yellow, "expand")
        println(" (belief) -> $(a) \t $(imm_r) \t $(upper)")
        act_node = ActionNode(a, bel, upper, imm_r, Vector{BeliefNode}())
        push!(bn.children, act_node)
    end
    @assert length(bn.children) == n_actions(pomdp)
end

"""
Expand actions (Add new belief nodes)
"""
function expand!{A}(an::ActionNode{A}, solver::MCVISolver, pomdp::POMDPs.POMDP)
    if !isempty(an.children)
        return nothing
    end
    for i in 1:solver.obs_branch # branching factor
        # Sample observation
        s = rand(solver.simulator.rng, an.belief)
        obs = generate_o(pomdp, nothing, nothing, s, solver.simulator.rng)
        bel = next(an.belief, obs, pomdp) # Next belief by observation

        upper = upperbound(bel, pomdp, solver.simulator.rng)
        lower = lowerbound(bel, pomdp, solver.simulator.rng)

        belief_node = BeliefNode(Nullable(obs), bel, upper, lower, Nullable{MCVINode}(), Vector{ActionNode{A}}())
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
    policy_node, node_val = backup(bn.belief, policy, solver.simulator, pomdp, solver.num_state,
                                   solver.num_prune_obs, solver.num_eval_belief, get(solver.scratch)) # Backup belief
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
# stack_size = 0
"""
Search over belief
"""
function search!{S,A,O}(bn::BeliefNode, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP{S,A,O}, target_gap::Float64)
    try
        println("belief -> $(get(bn.obs)) \t $(bn.upper) \t $(bn.lower)")
    catch e
        if isa(e, NullException)
            println("belief -> nothing \t $(bn.upper) \t $(bn.lower)")
        else
            throw(e)
        end
    end
    if (bn.upper - bn.lower) > target_gap
        # Add child action nodes to belief node
        expand!(bn, solver, pomdp)
        max_upper = -Inf
        local choice = Nullable{ActionNode{A}}()
        for ac in bn.children
            # Backup action
            backup!(ac, solver, pomdp)
            # Choose the one with max upper limit
            if max_upper < ac.upper
                max_upper = ac.upper
                choice = Nullable(ac)
            end
        end
        # global stack_size
        # stack_size += 1
        # println("=============== $stack_size ===============")
        # Seach over action
        search!(get(choice), solver, policy, pomdp, target_gap)
    end
    # backup belief
    backup!(bn, solver, policy, pomdp)
end

"""
Search over action
"""
function search!(an::ActionNode, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP, target_gap::Float64)
    println("act -> $(an.act) \t $(an.upper)")
    # if isterminal(pomdp, an.act) # FIXME Original MCVI searches until maxtime :( I could do that.
    #     return nothing
    # end
    # Expand action
    expand!(an, solver, pomdp)
    max_gap = 0.0
    local choice = Nullable{BeliefNode}()
    for b in an.children
        gap = b.upper - b.lower
        # Choose the belief that maximizes the gap bw upper and lower
        println("gap=$gap, maxgap=$max_gap")
        if gap > max_gap
            max_gap = gap
            choice = Nullable(b)
        end
    end
    # If we found anything that improved the difference
    if !isnull(choice)
        search!(get(choice), solver, policy, pomdp, target_gap/discount(pomdp))
    else
        println("Gap closed!")
    end
    # Backup action
    backup!(an, solver, pomdp)
end

"""
Solve function
"""
function solve(solver::MCVISolver, pomdp::POMDPs.POMDP, policy::MCVIPolicy=create_policy(solver, pomdp))
    if isnull(solver.root)
        initialize_root!(solver, pomdp)
    end
    # Gap between upper and lower
    target_gap = 0.0
    if policy.updater == nothing
        initialize_updater!(policy)
    end

    # Search
    for i in 1:solver.n_iter
        global stack_size
        stack_size = 0
        tic()
        search!(get(solver.root), solver, policy, pomdp, target_gap) # Here solver.root is a BeliefNode
        policy.updater.root = get(get(solver.root).best_node)             # Here policy.updater.root is a MCVINode

        if (get(solver.root).upper - get(solver.root).lower) < 0.1
            break
        end
        print_with_color(:green, "iter $(i) \t")
        println("upper: $(get(solver.root).upper) \t lower: $(get(solver.root).lower) \t time: $(toq())")

    end
    return policy
end
