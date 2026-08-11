using Test
using JSON
using HDF5

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

const O = MocosSimCMAESOptimizer
const TOLERANCE = 1e-10

function fixture_config(root::String, gt_dir::String)
    objective = O.ObjectiveConfig(
        Dict{String,Float64}(
            "daily_detections" => 1.0,
            "daily_deaths" => 0.5,
            "daily_detections_cumulative" => 0.25,
            "daily_deaths_cumulative" => 0.25,
            "weekly_control" => 0.0,
        ),
        2, 0.9, 0, "baseline", 0.0, 0.0,
    )
    posterior = O.PosteriorConfig(
        false, "diagonal_gaussian_weekly", 1, 1, 1,
        0.05, 1.0, 1.0, 1.0, 1.0, 0.0,
    )
    external = O.ExternalSimConfig(gt_dir, "fake-julia", root, "advanced_cli.jl", false)
    return O.OptimizerConfig(
        joinpath(root, "seed.json"), root, 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Tuple{Float64,Float64}}(),
        Dict{String,Dict{String,Any}}(), "monthly",
        Dict{String,Float64}("00_04" => 0.2, "05_14" => 0.3),
        Dict{String,Any}("enabled" => true, "holdout_days" => 3, "seeds" => [7]),
        objective, external, Dict{String,Vector{String}}(), nothing, posterior,
    )
end

function write_fixture_data(root::String)
    gt_dir = joinpath(root, "gt")
    mkpath(gt_dir)
    # Day 2 is missing in every reference series.  The simulated value at
    # day 2 is deliberately different, proving that it cannot be shifted
    # onto day 3 by either adapter.
    series = Dict(
        "daily_age_total_detections.csv" => [1.0, missing, 3.0, 4.0, 5.0, 6.0],
        "daily_hospitalizations.csv" => [2.0, missing, 4.0, 5.0, 6.0, 7.0],
        "daily_age_total_deaths.csv" => [0.0, missing, 1.0, 1.0, 2.0, 2.0],
        "sax-scholars-infections-normalized.csv" => [1.0, missing, 1.0, 1.0, 1.0, 1.0],
    )
    for (name, values) in series
        open(joinpath(gt_dir, name), "w") do io
            println(io, "day,value")
            for (day, value) in enumerate(values)
                println(io, value === missing ? "$day," : "$day,$value")
            end
        end
    end
    daily = joinpath(root, "output_daily.h5")
    h5open(daily, "w") do h5
        for (trajectory, offset) in (("trajectory_1", 0.0), ("trajectory_2", 0.2))
            group = create_group(h5, trajectory)
            write(group, "daily_detections", [1.0, 99.0, 3.0, 4.0, 5.0, 6.0] .+ offset)
            write(group, "daily_hospitalizations", [2.0, 99.0, 4.0, 5.0, 6.0, 7.0] .+ offset)
            write(group, "daily_deaths", [0.0, 99.0, 1.0, 1.0, 2.0, 2.0] .+ offset)
            write(group, "daily_age_total_detections", [1.0, 99.0, 3.0, 4.0, 5.0, 6.0] .+ offset)
            write(group, "daily_age_total_deaths", [0.0, 99.0, 1.0, 1.0, 2.0, 2.0] .+ offset)
        end
    end
    return gt_dir, daily
end

function candidate_fixture()
    return Dict{String,Any}(
        "infection_modulation" => Dict{String,Any}(
            "params" => Dict{String,Any}("interval_values" => [0.2, 0.25, 0.3]),
        ),
    )
end

function canonical_metric_payload(cfg, daily, days)
    gt = O.load_gt_series(cfg.external_sim.gt_dir)
    periods = Dict{String,Any}()
    retained = Dict{String,Any}()
    for metric in sort(collect(keys(gt)))
        isempty(gt[metric]) && continue
        weekly = O.weekly_error_distributions(daily, metric, gt[metric], days)
        periods[metric] = weekly["periods"]
        retained[metric] = weekly["retained_indices"]
    end
    score, metrics = O.score_from_daily(cfg, daily, days, candidate_fixture())
    return Dict{String,Any}(
        "status" => "completed",
        "failure_class" => nothing,
        "score" => score,
        "metrics" => metrics,
        "metric_manifest" => metrics["effective_metric_manifest"],
        "periods" => periods,
        "retained_indices" => retained,
        "holdout" => metrics["validation_window"],
    )
end

function marker_status(root::String)
    done = isfile(joinpath(root, "done.ok"))
    failed = isfile(joinpath(root, "failed.ok"))
    skipped = isfile(joinpath(root, "skipped.ok"))
    # This is the collection contract: terminal failure beats a stale done
    # marker, while done beats skipped when no failure marker exists.
    if failed
        return "failed", "process_failure"
    elseif done
        return "completed", nothing
    elseif skipped
        return "skipped", "skipped_by_grace_period"
    end
    return "failed", "missing_terminal_marker"
end

function local_adapter(root::String, cfg, days::Int)
    status, failure = marker_status(root)
    daily = joinpath(root, "output_daily.h5")
    if status == "completed" && !isfile(daily)
        return Dict{String,Any}(
            "status" => "failed", "failure_class" => "missing_output",
            "score" => Inf, "metrics" => Dict{String,Any}(),
        )
    elseif status != "completed"
        return Dict{String,Any}(
            "status" => status, "failure_class" => failure,
            "score" => Inf, "metrics" => Dict{String,Any}(),
        )
    end
    try
        payload = canonical_metric_payload(cfg, daily, days)
        payload["root_role"] = "local"
        return payload
    catch
        return Dict{String,Any}(
            "status" => "failed", "failure_class" => "score_exception",
            "score" => Inf, "metrics" => Dict{String,Any}(),
        )
    end
end

function collected_adapter(root::String, cfg, days::Int)
    list_path = joinpath(root, "candidate_list.txt")
    listed = isfile(list_path) ? [strip(x) for x in readlines(list_path) if !isempty(strip(x))] : String[]
    # Reconstruction is intentionally based on the candidate list, not on
    # directory enumeration, matching a collected/Slurm controller seam.
    root in listed || return Dict{String,Any}(
        "status" => "failed", "failure_class" => "candidate_not_collected",
        "score" => Inf, "metrics" => Dict{String,Any}(),
    )
    payload = local_adapter(root, cfg, days)
    payload["root_role"] = "collected"
    return payload
end

function strip_transport(payload)
    result = deepcopy(payload)
    pop!(result, "root_role", nothing)
    return result
end

function assert_numeric_equal(a, b)
    if a isa Number && b isa Number
        (isinf(a) || isinf(b)) ? (@test a == b) : (@test isapprox(a, b; atol=TOLERANCE, rtol=TOLERANCE))
    elseif a isa AbstractDict && b isa AbstractDict
        @test sort(String.(collect(keys(a)))) == sort(String.(collect(keys(b))))
        for key in keys(a)
            assert_numeric_equal(a[key], b[key])
        end
    elseif a isa AbstractVector && b isa AbstractVector
        @test length(a) == length(b)
        for (x, y) in zip(a, b)
            assert_numeric_equal(x, y)
        end
    else
        @test a == b
    end
end

@testset "deterministic collected scoring parity" begin
    mktempdir() do root
        gt_dir, daily = write_fixture_data(root)
        cfg = fixture_config(root, gt_dir)
        candidate_root = joinpath(root, "candidate_001")
        mkpath(candidate_root)
        cp(daily, joinpath(candidate_root, "output_daily.h5"))
        open(joinpath(candidate_root, "candidate.json"), "w") do io
            JSON.print(io, candidate_fixture())
        end
        touch(joinpath(candidate_root, "done.ok"))
        open(joinpath(root, "candidate_list.txt"), "w") do io
            println(io, candidate_root)
        end
        open(joinpath(candidate_root, "candidate_list.txt"), "w") do io
            println(io, candidate_root)
        end

        local_payload = local_adapter(candidate_root, cfg, 6)
        collected_payload = collected_adapter(candidate_root, cfg, 6)
        @test local_payload["status"] == "completed"
        @test collected_payload["status"] == "completed"
        @test local_payload["failure_class"] === nothing
        assert_numeric_equal(strip_transport(local_payload), strip_transport(collected_payload))
        @test local_payload["holdout"]["requested_start_day"] == 4
        @test local_payload["holdout"]["retained_indices"] == [4, 5, 6]
        @test local_payload["periods"]["daily_detections"] == [[1, 6]]
        @test local_payload["retained_indices"]["daily_detections"] == [[1, 3, 4, 5, 6]]
        @test haskey(local_payload["metric_manifest"], "weekly_control")
        @test local_payload["metric_manifest"]["weekly_control"]["weight"] == 0.0
        @test isfinite(local_payload["score"])
    end
end

@testset "collected marker and failure parity" begin
    mktempdir() do root
        gt_dir, daily = write_fixture_data(root)
        cfg = fixture_config(root, gt_dir)
        cases = (
            ("marker_conflict", ["done.ok", "failed.ok"], nothing, "process_failure"),
            ("missing_output", ["done.ok"], nothing, "missing_output"),
            ("failed", ["failed.ok"], daily, "process_failure"),
            ("skipped", ["skipped.ok"], daily, "skipped_by_grace_period"),
        )
        for (name, markers, output, expected_failure) in cases
            case_root = joinpath(root, name)
            mkpath(case_root)
            output === nothing || cp(output, joinpath(case_root, "output_daily.h5"))
            for marker in markers
                touch(joinpath(case_root, marker))
            end
            open(joinpath(case_root, "candidate_list.txt"), "w") do io
                println(io, case_root)
            end
            local_payload = local_adapter(case_root, cfg, 6)
            collected_payload = collected_adapter(case_root, cfg, 6)
            assert_numeric_equal(strip_transport(local_payload), strip_transport(collected_payload))
            @test local_payload["status"] != "completed"
            @test local_payload["failure_class"] == expected_failure
            @test isinf(local_payload["score"])
        end
    end
end
