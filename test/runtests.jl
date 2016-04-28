using MCVI
using Base.Test
import MCVI: lowerbound, upperbound, init_lower_action

include("LightDark.jl")

function test_LightDark1D()
    p = LightDark1D()
    @test discount(p) == 0.9
    s0 = LightDark1DState(0,0)
    low0 = lowerbound(p, s0)
    @test low0 == 9.0
    s1, r = generate_sr(p, s0, +1, p.rng)
    @test s1.y == 1.0
    @test r == 0.0
    s2, r = generate_sr(p, s1, 0, p.rng)
    @test s2.x != 0
    @test r == -10
    s3 = LightDark1DState(0, 5)
    obs = generate_o(p, nothing, nothing, s3, p.rng)
    @test abs(obs-5.0) <= 0.1
    return true
end

@test test_LightDark1D()

include("test_policy.jl")

include("test_updater.jl")

include("test_belief.jl")

@test test_dummy_graph()
@test test_dummy_graph2()

@test test_backup()

@test test_belief()

include("test_solve.jl")
@test test_solve()