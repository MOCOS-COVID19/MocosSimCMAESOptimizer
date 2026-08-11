using Test
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

const O = MocosSimCMAESOptimizer

@testset "paired scoring retains original indices" begin
    gt = Union{Missing,Float64}[1.0, missing, 3.0, 4.0]
    sim = [1.0, 99.0, 3.0]
    g, s, idx = O.paired_observations(gt, sim, 4)
    @test idx == [1, 3]
    @test g == [1.0, 3.0]
    @test s == [1.0, 3.0]
    @test O.paired_observations(gt, Float64[], 4)[3] == Int[]
end

@testset "nonfinite observations are omitted symmetrically" begin
    g, s, idx = O.paired_observations([1.0, NaN, 3.0], [1.0, 2.0, Inf], 3)
    @test idx == [1]
    @test g == [1.0] && s == [1.0]
end

@testset "required objective inputs cannot disappear" begin
    objective = O.ObjectiveConfig(Dict{String,Float64}("daily_detections" => 1.0),
        1, 1.0, 1, "baseline", 0.0, 0.0)
    posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
    cfg = O.OptimizerConfig("seed", "out", 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(), Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}(), objective, nothing, Dict{String,Vector{String}}(),
        nothing, posterior)
    @test isinf(O.objective_score(cfg, Dict{String,Float64}("daily_detections" => Inf),
        0.0, 0.0, 0.0))
    @test isfinite(O.objective_score(cfg, Dict{String,Float64}(), 0.0, 0.0, 0.0))
end
