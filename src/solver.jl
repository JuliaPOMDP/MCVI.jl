function action(policy::MCVIPolicy, node::MCVINode)
    return node.act             # Confused now. What should be the second argument again?
end

abstract type TreeNode end

mutable struct BeliefNode{O} <: TreeNode
    obs::Union{O, Nothing}
    belief:: MCVIBelief
    upper::Reward
    lower::Reward
    best_node::Union{MCVINode, Nothing}
    children::Vector{TreeNode}
end

mutable struct ActionNode{O,A} <: TreeNode
    act::A
    belief:: MCVIBelief
    upper::Reward
    imm_reward::Reward
    children::Vector{BeliefNode{O}}
end

#BeliefNode{O}(obs::Union{O, Nothing}, b::MCVIBelief, u::Reward, l::Reward, bn::Union{MCVINode, Nothing}, c::Vector{ActionNode{O,A}}) where {O,A} = BeliefNode(obs, b, u, l, bn, c)

"""

Hyperparameters:

- `n_iter`          : Number of iterations
- `num_particles`   : Number of belief particles to be used
- `obs_branch`      : Branching factor [default 8?]
- `num_state`       : Number of states to sample from belief [default 500?]
- `num_prune_obs`   : Number of times to sample observation while pruning alpha edges [default 1000?]
- `num_eval_belief` : Number of times to simulate while evaluating belief [default 5000?]
- `num_obs`         : [default 50?]

Bounds:

- `lbound`          : An object representing the lower bound. The function `MCVI.lower_bound(lbound, problem, s)` will be called to get the lower bound for the state `s` - this function needs to be implemented for the solver to work.
- `ubound`          : An object representing the upper bound. The function `MCVI.upper_bound(ubound, problem, s)` will be called to get the lower bound for the state `s` - this function needs to be implemented for the solver to work.

See `$(joinpath(dirname(pathof(MCVI)),"..", "test","runtests.jl"))` for an example of bounds implemented for the Light Dark problem.
"""
mutable struct MCVISolver <: POMDPs.Solver
    simulator::POMDPs.Simulator
    root::Union{BeliefNode, Nothing}
    n_iter::Int64
    num_particles::Int64
    obs_branch::Int64
    num_state::Int64
    num_prune_obs::Int64
    num_eval_belief::Int64

    num_obs::Int64
    lbound::Any
    ubound::Any
    scratch::Union{Scratch, Nothing}
    function MCVISolver(sim, root, n_iter, nbp, ob, ns, npo, neb, num_obs, lb, ub)
        new(sim, root, n_iter, nbp, ob, ns, npo, neb, num_obs, lb, ub, nothing)
    end
end

function initialize_root!(solver::MCVISolver, pomdp::POMDPs.POMDP{S,A,O}) where {S,A,O}
    b0 = initial_belief(pomdp, solver.num_particles, solver.simulator.rng)
    solver.root = BeliefNode{O}(nothing, b0, upper_bound(solver.ubound, pomdp, b0), lower_bound(solver.lbound, pomdp, b0), nothing, Vector{TreeNode}())
    solver.scratch = Scratch(Vector{O}(undef, solver.num_obs), zeros(solver.num_obs), zeros(solver.num_obs), zeros(solver.num_obs, 2))
end

create_policy(::MCVISolver, p::POMDPs.POMDP) = MCVIPolicy(p)

"""
Expand beliefs (Add new action nodes)
"""
function expand!(bn::BeliefNode{O}, solver::MCVISolver, pomdp::POMDPs.POMDP; debug=false) where {O}
    if !isempty(bn.children)
        return nothing
    end

    as = actions(pomdp)
    for a in as
        bel = next(bn.belief, a, pomdp, solver.simulator.rng) # Next belief by action
        imm_r = reward(bel, pomdp)
        local upper::Float64
        if isterminal(pomdp, a) # FIXME This is  necessary?
            upper = imm_r*discount(pomdp)
        else
        # Initialize using problem upper value
            upper = upper_bound(solver.ubound, pomdp, bel)
        end
        debug && printstyled("expand", color=:yellow)
        debug && println(" (belief) -> $(a) \t $(imm_r) \t $(upper)")
        act_node = ActionNode(a, bel, upper, imm_r, Vector{BeliefNode{O}}())
        push!(bn.children, act_node)
    end
    @assert length(bn.children) == length(as)
end

"""
Expand actions (Add new belief nodes)
"""
function expand!(an::ActionNode{O,A}, solver::MCVISolver, pomdp::POMDPs.POMDP; debug=false) where {O,A}
    if !isempty(an.children)
        return nothing
    end
    for i in 1:solver.obs_branch # branching factor
        # Sample observation
        s = rand(solver.simulator.rng, an.belief)
        obs = gen(DDNNode(:o), pomdp, s, solver.simulator.rng)
        bel = next(an.belief, obs, pomdp) # Next belief by observation

        upper = upper_bound(solver.ubound, pomdp, bel)
        lower = lower_bound(solver.lbound, pomdp, bel)

        belief_node = BeliefNode{O}(obs, bel, upper, lower, nothing, Vector{ActionNode{O,A}}())
        push!(an.children, belief_node)
    end
end

"""
Backup over belief
"""
function backup!(bn::BeliefNode{O}, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP; debug=false) where {O}
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
                                   solver.num_prune_obs, solver.num_eval_belief, solver.scratch, debug=debug) # Backup belief
    debug && printstyled("backup", color=:magenta)
    debug && println(" (belief) -> $(node_val) \t $(bn.lower)")
    if node_val > bn.lower
        bn.lower = node_val
        bn.best_node = policy_node
        addnode!(policy.updater, policy_node) # Add node to policy graph
    end
end

"""
Backup over action
"""
function backup!(an::ActionNode{O,A}, solver::MCVISolver, pomdp::POMDPs.POMDP) where {O,A}
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
function search!(bn::BeliefNode{O}, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP{S,A,O}, target_gap::Float64; debug=false) where {S,A,O}
    debug && println("belief -> $(bn.obs) \t $(bn.upper) \t $(bn.lower)")
    if (bn.upper - bn.lower) > target_gap
        # Add child action nodes to belief node
        expand!(bn, solver, pomdp, debug=debug)
        max_upper = -Inf
        local choice = nothing
        for ac in bn.children
            # Backup action
            backup!(ac, solver, pomdp)
            # Choose the one with max upper limit
            if max_upper < ac.upper
                max_upper = ac.upper
                choice = ac
            end
        end
        # global stack_size
        # stack_size += 1
        # println("=============== $stack_size ===============")
        # Seach over action
        search!(choice, solver, policy, pomdp, target_gap, debug=debug)
    end
    # backup belief
    backup!(bn, solver, policy, pomdp)
end

"""
Search over action
"""
function search!(an::ActionNode{O,A}, solver::MCVISolver, policy::MCVIPolicy, pomdp::POMDPs.POMDP, target_gap::Float64; debug=false) where {O,A}
    debug && println("act -> $(an.act) \t $(an.upper)")
    if isterminal(pomdp, an.act) # FIXME Original MCVI searches until maxtime :( I could do that.
        return nothing
    end
    # Expand action
    expand!(an, solver, pomdp, debug=debug)
    max_gap = 0.0
    local choice = nothing
    for b in an.children
        gap = b.upper - b.lower
        # Choose the belief that maximizes the gap bw upper and lower
        debug && println("gap=$gap, maxgap=$max_gap")
        if gap > max_gap
            max_gap = gap
            choice = b
        end
    end
    # If we found anything that improved the difference
    if choice == nothing
        println("Gap closed!")
    else
        search!(choice, solver, policy, pomdp, target_gap/discount(pomdp), debug=debug)
    end
    # Backup action
    backup!(an, solver, pomdp)
end

"""
Solve function
"""
function solve(solver::MCVISolver, pomdp::POMDPs.POMDP{S,A,O}, policy::MCVIPolicy=create_policy(solver, pomdp); debug=false) where {S,A,O}
    if solver.root == nothing
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
        t = @elapsed begin
            search!(solver.root, solver, policy, pomdp, target_gap, debug=debug) # Here solver.root is a BeliefNode
            policy.updater.root = solver.root.best_node             # Here policy.updater.root is a MCVINode

            if (solver.root.upper - solver.root.lower) < 0.1
                break
            end
        end
        debug && printstyled("iter $(i) \t", color=:green)
        debug && println("upper: $(solver.root.upper) \t lower: $(solver.root.lower) \t time: $(t)")
    end

    if @implemented initialstate_distribution(::typeof(pomdp))
        policy.updater.root_belief = initialstate_distribution(pomdp)
    else
        policy.updater.root_belief = nothing
    end

    return policy
end


@POMDP_require solve(solver::MCVISolver, pomdp::POMDP) begin
    P = typeof(pomdp)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    LB = typeof(solver.lbound)
    UB = typeof(solver.ubound)
    @req actions(::P)
    as = actions(pomdp)
    @req length(::typeof(as))
    @req generate_sr(::P, ::S, ::A, ::AbstractRNG)
    @req generate_o(::P, ::S, ::A, ::S, ::AbstractRNG)
    @req initialstate(::P, ::AbstractRNG)
    @req lower_bound(::LB, ::P, ::S)
    @req upper_bound(::UB, ::P, ::S)
    @req init_lower_action(::P)
    @req isterminal(::P, ::A)
    @req isterminal(::P, ::S)
    @req discount(::P)
end
