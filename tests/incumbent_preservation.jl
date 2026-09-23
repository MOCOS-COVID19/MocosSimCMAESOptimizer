using Test
using MocosSimCMAESOptimizer

@testset "incumbent population preservation" begin
    candidates = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    steps = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    incumbent = [7.0, 8.0]

    slot = preserve_incumbent!(candidates, steps, incumbent)

    @test slot == 3
    @test candidates[slot] == incumbent
    @test steps[slot] == [0.0, 0.0]
    @test candidates[1] == [1.0, 2.0]
    @test_throws ArgumentError preserve_incumbent!(Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), incumbent)
    @test_throws ArgumentError preserve_incumbent!([[1.0]], [[0.0]], incumbent)
end
