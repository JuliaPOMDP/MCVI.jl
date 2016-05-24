function test_belief()
    p = LightDark1D()
    ss0 = MCVI.MCVISubspace([LightDark1DState(0, 4), LightDark1DState(0, 1)], Float64[], Dict{Int64, MCVI.MCVISubspace{LightDark1DState,Int64}}())
    b0 = MCVI.MCVIBelief(ss0, [0.5, 0.5], UInt64(1))
    b1 = MCVI.next(b0, 1, p, MersenneTwister(42))
    @test MCVI.particle(b1.space, 1).y == 5
    b2 = MCVI.next(b0, 5.0, p)
    @test !isnan(b2.weights[1])
    @test b2.weights[1] >= b2.weights[2]
    return true
end
