using Test
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using HDF5
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

@testset "disabled infinite metrics do not poison objective" begin
    objective = O.ObjectiveConfig(Dict{String,Float64}(
            "daily_detections" => 1.0,
            "daily_student_detections" => 0.0,
        ), 1, 1.0, 1, "baseline", 0.0, 0.0)
    posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
    cfg = O.OptimizerConfig("seed", "out", 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(), Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}(), objective, nothing, Dict{String,Vector{String}}(),
        nothing, posterior)
    metrics = Dict{String,Any}(
        "daily_detections" => 2.0,
        "daily_student_detections" => Inf,
    )
    @test O.objective_score(cfg, metrics, 0.0, 0.0, 0.0) == 2.0
end

@testset "zero-weight nonfinite optional terms are ignored" begin
    objective = O.ObjectiveConfig(Dict{String,Float64}(
            "daily_detections" => 1.0,
            "weekly_control" => 0.0,
        ), 1, 1.0, 1, "baseline", 0.0, 0.0)
    posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
    cfg = O.OptimizerConfig("seed", "out", 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(), Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}(), objective, nothing, Dict{String,Vector{String}}(),
        nothing, posterior)
    @test O.objective_score(cfg, Dict{String,Any}("daily_detections" => 2.0),
        Inf, Inf, Inf) == 2.0
end

@testset "positive-weight nonfinite optional terms propagate Inf" begin
    objective = O.ObjectiveConfig(Dict{String,Float64}(
            "daily_detections" => 1.0,
            "weekly_control" => 1.0,
        ), 1, 1.0, 1, "baseline", 0.5, 0.25)
    posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
    cfg = O.OptimizerConfig("seed", "out", 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(), Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}(), objective, nothing, Dict{String,Vector{String}}(),
        nothing, posterior)
    @test isinf(O.objective_score(cfg, Dict{String,Any}("daily_detections" => 2.0),
        Inf, 0.0, 0.0))
    @test isinf(O.objective_score(cfg, Dict{String,Any}("daily_detections" => 2.0),
        0.0, Inf, 0.0))
    @test isinf(O.objective_score(cfg, Dict{String,Any}("daily_detections" => 2.0),
        0.0, 0.0, Inf))
end

@testset "effective metric manifest records weighted optional terms" begin
    objective = O.ObjectiveConfig(Dict{String,Float64}(
            "daily_detections" => 1.0,
            "weekly_control" => 0.0,
        ), 1, 1.0, 1, "baseline", 0.0, 0.25)
    posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
    cfg = O.OptimizerConfig("seed", "out", 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(), Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}(), objective, nothing, Dict{String,Vector{String}}(),
        nothing, posterior)
    manifest = O.effective_metric_manifest(
        cfg, Dict{String,Any}("daily_detections" => 2.0), Inf, Inf, 0.0)
    @test manifest["weekly_control"]["weight"] == 0.0
    @test manifest["weekly_control"]["enabled"] == false
    @test manifest["temporal_jump_penalty"]["weight"] == 0.0
    @test manifest["infection_extrema_penalty"]["weight"] == 0.25
    @test manifest["infection_extrema_penalty"]["enabled"] == true
    @test manifest["weekly_control"]["source_present"] == true
end

@testset "cumulative distribution guards empty and preserves indices" begin
    mktempdir() do root
        daily = joinpath(root, "daily.h5")
        h5open(daily, "w") do h5
            grp = create_group(h5, "trajectory_1")
            write(grp, "daily_detections", [1.0, 99.0, 3.0])
        end
        @test O.cumulative_error_distribution(
            daily, "daily_detections", Union{Missing,Float64}[], 3) == Float64[]
        values = O.cumulative_error_distribution(
            daily, "daily_detections",
            Union{Missing,Float64}[1.0, missing, 3.0], 3)
        @test length(values) == 1
        @test values[1] == 0.0
    end
end

@testset "daily scoring returns heterogeneous validation diagnostics" begin
    mktempdir() do root
        gt_dir = joinpath(root, "gt")
        mkpath(gt_dir)
        for (name, values) in (
            ("daily_age_total_detections.csv", [1.0, 2.0, 3.0]),
            ("daily_hospitalizations.csv", [1.0, 2.0, 3.0]),
            ("daily_age_total_deaths.csv", [1.0, 2.0, 3.0]),
            ("sax-scholars-infections-normalized.csv", [1.0, 2.0, 3.0]),
        )
            open(joinpath(gt_dir, name), "w") do io
                println(io, "day,value")
                for (day, value) in enumerate(values)
                    println(io, "$day,$value")
                end
            end
        end
        daily = joinpath(root, "daily.h5")
        h5open(daily, "w") do h5
            grp = create_group(h5, "trajectory_1")
            for metric in ("daily_detections", "daily_hospitalizations",
                           "daily_deaths", "daily_age_total_detections",
                           "daily_age_total_deaths")
                write(grp, metric, [1.0, 2.0, 3.0])
            end
        end
        ext = O.ExternalSimConfig(gt_dir, "julia", root, joinpath(root, "unused.jl"), false)
        objective = O.ObjectiveConfig(Dict{String,Float64}(
            "daily_detections" => 1.0, "daily_deaths" => 1.0,
            "weekly_control" => 0.0), 1, 1.0, 1, "baseline", 0.0, 0.0)
        posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
            0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
        cfg = O.OptimizerConfig("seed", root, 30, O.StageConfig[],
            Dict{String,Tuple{Float64,Float64}}(), Dict{String,Tuple{Float64,Float64}}(),
            Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
            Dict{String,Any}("enabled" => true, "holdout_days" => 2),
            objective, ext, Dict{String,Vector{String}}(), nothing, posterior)
        score, payload = O.score_from_daily(cfg, daily, 3)
        @test isfinite(score)
        @test payload["validation_window"] isa Dict{String,Any}
        @test payload["effective_metric_manifest"] isa Dict{String,Any}
        @test payload["effective_metric_manifest"]["weekly_control"]["weight"] == 0.0
    end
end
