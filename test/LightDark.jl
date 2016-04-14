import Base: ==, +, *, -

using POMDPs
import POMDPs: create_state, discount, isterminal, pdf, actions, iterator, n_actions
using GenerativeModels
import GenerativeModels: generate_sr, generate_o, initial_state

type LightDark1DState
    x::Float64
    y::Float64
    LightDark1DState() = new()
    LightDark1DState(x, y) = new(x, y)
end
==(s1::LightDark1DState, s2::LightDark1DState) = (s1.x == s2.x) && (s1.y == s2.y)
+(s1::LightDark1DState, s2::LightDark1DState) = LightDark1DState(s1.x+s2.x, s1.y+s2.y)
-(s1::LightDark1DState, s2::LightDark1DState) = LightDark1DState(s1.x-s2.x, s1.y-s2.y)
*(n::Number, s::LightDark1DState) = LightDark1DState(n*s.x, n*s.y)
*(s1::LightDark1DState, s2::LightDark1DState) = LightDark1DState(s1.x*s2.x, s1.y*s2.y)

Base.hash(s::LightDark1DState, h::UInt64=zero(UInt64)) = hash(s.x, hash(s.y, h))
copy(s::LightDark1DState) = LightDark1DState(s.x, s.y)

type LightDark1D <: POMDPs.POMDP{LightDark1DState,Int64,Float64}
    discount_factor::Float64
    # lower_act::Int64
    rng::AbstractRNG

    step::Float64
    movement_cost::Float64
end
LightDark1D() = LightDark1D(0.9, MersenneTwister(42), 1, 0)

create_state(p::LightDark1D) = LightDark1DState(0,0)

discount(p::LightDark1D) = p.discount_factor

isterminal(::LightDark1D, act::Int64) = act == 0

function initial_state(p::LightDark1D)
    return LightDark1DState(0, 2+Base.randn(p.rng)*3)
end

function generate_sr{LightDark1DState}(p::LightDark1D, s::LightDark1DState, a::Int64, rng::AbstractRNG)
    sprime = copy(s)
    if sprime.x > 0
        r = 0
        return (sprime, r)
    end
    if a == 0
        sprime.x = 1.0
        if abs(sprime.y) < 1
            r = 10
        else
            r = -10
        end
    else
        sprime.y += a
        r = 0
    end
    return (sprime, r)
end

sigma(x::Float64) = abs(x - 5)/sqrt(2) + 1e-2
function generate_o(p::LightDark1D, s, a, sp::LightDark1DState, rng::AbstractRNG)
    return sp.y + Base.randn(p.rng)*sigma(sp.y)
end

function init_lower_action(p::LightDark1D)
    return 0 # p.lower_act
end

function lowerbound(p::LightDark1D, s::LightDark1DState)
    _, r = generate_sr(p, s, init_lower_action(p), p.rng)
    return r * discount(p)
end

function upperbound(p::LightDark1D, s::LightDark1DState)
    steps = abs(s.y)/p.step + 1
    return 10*(discount(p)^steps)
end

gauss(s::Float64, x::Float64) = 1 / sqrt(2*pi) / s * exp(-1*x^2/(2*s^2))
function pdf(s::LightDark1DState, obs::Float64)
    return gauss(sigma(s.y), s.y-obs)
end

type LightDark1DActionSpace <: POMDPs.AbstractSpace
    actions::Vector{Int64}
end
Base.length(asp::LightDark1DActionSpace) = length(asp.actions)
actions(::LightDark1D) = LightDark1DActionSpace([-1, 0, 1]) # TODO
actions(pomdp::LightDark1D, s::LightDark1DState, acts::LightDark1DActionSpace=actions(pomdp)) = acts
iterator(space::LightDark1DActionSpace) = space.actions
dimensions(::LightDark1DActionSpace) = 1
n_actions(p::LightDark1D) = length(actions(p))

function rand(rng::AbstractRNG, asp::LightDark1DActionSpace, a::Int64)
    a = rand(rng, iterator(asp))
    return a
end

# Define some simple policies based on particle belief

type DummyHeuristic1DPolicy <: POMDPs.Policy
    thres::Float64
end
DummyHeuristic1DPolicy() = DummyHeuristic1DPolicy(0.1)

type SmartHeuristic1DPolicy <: POMDPs.Policy
    thres::Float64
end
SmartHeuristic1DPolicy() = SmartHeuristic1DPolicy(0.1)

function action{B}(p::DummyHeuristic1DPolicy, b::B)
    target = 0.0
    μ = mean(b)
    σ = std(b, μ)

    if σ.y < p.thres && -0.5 < μ.y < 0.5
        a = 0
    elseif μ.y < target
        a = 1                   # Right
    elseif μ.y > target
        a = -1                  # Left
    end
    return a
end

function action{B}(p::SmartHeuristic1DPolicy, b::B)
    μ = mean(b)
    σ = std(b, μ)
    target = 0.0
    if σ.y > p.thres
        target = 5.0
    end
    if σ.y < p.thres && -0.5 < μ.y < 0.5
        a = 0
    elseif μ.y < target
        a = 1                   # Right
    elseif μ.y > target
        a = -1                  # Left
    end
    return a
end
