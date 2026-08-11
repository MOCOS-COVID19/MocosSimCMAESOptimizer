using Test
using HDF5
using JSON
using Random

include(joinpath(@__DIR__, "..", "src", "MocosSimCMAESOptimizer.jl"))
const O = MocosSimCMAESOptimizer

function local_stage_fixture(; fail_first::Bool)
    root = mktempdir()
    gt = joinpath(root, "gt")
    mkpath(gt)
    for name in ("daily_age_total_detections.csv",
                 "daily_hospitalizations.csv",
                 "daily_age_total_deaths.csv",
                 "sax-scholars-infections-normalized.csv")
        write(joinpath(gt, name), "day,value\n1,1\n2,1\n")
    end
    source_daily = joinpath(root, "source_daily.jld2")
    h5open(source_daily, "w") do h5
        trajectory = create_group(h5, "trajectory_1")
        for metric in ("daily_detections", "daily_hospitalizations",
                       "daily_deaths")
            write(trajectory, metric, [1.0, 1.0])
        end
    end
    fake = joinpath(root, "fake_launcher.sh")
    first_policy = fail_first ? "case \"\$out\" in *cand_01/*) exit 7;; esac\n" : ""
    write(fake, "#!/bin/sh\nout=\"\"\nprev=\"\"\nfor arg in \"\$@\"; do\n" *
        "  if [ \"\$prev\" = \"--output-daily\" ]; then out=\"\$arg\"; fi\n" *
        "  prev=\"\$arg\"\ndone\n" * first_policy *
        "cp " * source_daily * " \"\$out\"\nexit 0\n")
    chmod(fake, 0o755)
    objective = O.ObjectiveConfig(
        Dict{String,Float64}("daily_detections" => 1.0,
                             "daily_hospitalizations" => 1.0,
                             "daily_deaths" => 1.0,
                             "weekly_control" => 0.0),
        2, 0.9, 0, "baseline", 0.0, 0.0)
    posterior = O.PosteriorConfig(false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0)
    cfg = O.OptimizerConfig(
        joinpath(root, "seed.json"), root, 1,
        [O.StageConfig("fixture", 2, 1, 10, 0.05)],
        Dict{String,Tuple{Float64,Float64}}("x" => (0.0, 1.0)),
        Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}("enabled" => false), objective,
        O.ExternalSimConfig(gt, fake, root, joinpath(root, "unused.jl"), false),
        Dict{String,Vector{String}}(), nothing, posterior)
    return root, cfg
end

function run_local_fixture(; fail_first::Bool)
    root, cfg = local_stage_fixture(; fail_first=fail_first)
    seed = Dict{String,Any}("x" => 0.5)
    stage = cfg.stages[1]
    specs = [O.ParamSpec("x", :scalar, 1, 0.0, 1.0)]
    result, _ = O.run_stage(MersenneTwister(17), seed, specs, cfg, stage, nothing;
        use_slurm=false)
    metrics_path = joinpath(root, "real_sims", "fixture", "iter_1", "candidate_list.txt")
    rows = readlines(joinpath(root, "real_sims", "fixture", "iter_metrics.jsonl"))
    return result, rows, metrics_path
end

@testset "fresh-root local run_stage terminal parity" begin
    result9, rows9, _ = run_local_fixture(; fail_first=true)
    @test length(rows9) == 10
    @test all(JSON.parse(row)["threshold_reached"] for row in rows9)
    @test all(!JSON.parse(row)["iteration_truncated"] for row in rows9)
    @test count(JSON.parse(row)["status"] == "completed" for row in rows9) == 9

    result10, rows10, _ = run_local_fixture(; fail_first=false)
    @test length(rows10) == 10
    @test all(JSON.parse(row)["threshold_reached"] for row in rows10)
    @test all(!JSON.parse(row)["iteration_truncated"] for row in rows10)
    @test count(JSON.parse(row)["status"] == "completed" for row in rows10) == 10
end
