using Test
using JSON
using SHA

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function production_config(root)
    cfgdir = joinpath(root, "config")
    mkpath(cfgdir)
    seed = Dict("infection_modulation" => Dict("params" => Dict(
        "interval_times" => [30, 60], "interval_values" => [0.2, 0.2])))
    O.safe_save_json(joinpath(cfgdir, "seed.json"), seed)
    raw = Dict{String,Any}(
        "seed_config" => "seed.json", "output_dir" => joinpath(root, "output"),
        "monthly_days" => 30,
        "stages" => [Dict("name" => "short", "fit_months" => 3, "max_iterations" => 1,
                          "population_size" => 1, "sigma" => 0.1),
                     Dict("name" => "long", "fit_months" => 6, "max_iterations" => 1,
                          "population_size" => 1, "sigma" => 0.1)],
        "scalar_bounds" => Dict("x" => [0.0, 1.0]), "temporal_bounds" => Dict{String,Any}(),
        "objective" => Dict("weights" => Dict("daily_detections" => 1.0)),
        "validation" => Dict("enabled" => false), "age_population_weights" => Dict("all" => 1.0),
        "posterior" => Dict("enabled" => false))
    path = joinpath(cfgdir, "config.json")
    O.safe_save_json(path, raw)
    return O.load_config(path)
end

function write_production_predecessor(root)
    stage_root = joinpath(root, "real_sims", "short")
    iter_root = joinpath(stage_root, "iter_1")
    mkpath(iter_root)
    names = ["x[1]", "x[2]"]
    values = [0.2, 0.4]
    trajectory = Dict("identity" => "trajectory-production-fixture",
                      "values" => values, "prefix_values" => values)
    prefix_hash = bytes2hex(SHA.sha256(JSON.json(values)))
    locks = [Dict("name" => names[i], "start_day" => i, "end_day" => i,
                  "value" => values[i], "class" => "locked") for i in 1:2]
    cma = Dict("parameter_names" => names, "mean" => values, "sigma" => [0.1, 0.1],
               "covariance" => [1.0 0.0; 0.0 1.0], "p_c" => [0.0, 0.0],
               "p_sigma" => [0.0, 0.0])
    candidate = Dict{String,Any}("candidate" => "1", "score" => 1.0,
        "status" => "completed", "stage" => "short", "iteration" => 1,
        "fit_months" => 3, "parameter_names" => names, "evaluated_vector" => values)
    archive = [candidate]
    O.safe_save_json(joinpath(stage_root, "survivor_archive.json"), archive)
    O.persist_archive_transfer_manifest(stage_root, archive; stage="short", fit_months=3)
    O.safe_save_json(joinpath(stage_root, "top_candidates.json"), archive)
    open(joinpath(stage_root, "iter_metrics.jsonl"), "w") do io
        println(io, JSON.json(candidate))
    end
    O.safe_save_json(joinpath(iter_root, "top_candidates.json"), archive)
    O.safe_save_json(joinpath(iter_root, "cma_sampling_state.json"), Dict("state" => "fixture"))
    open(joinpath(iter_root, "candidate_list.txt"), "w") do io
        println(io, "short-a")
    end
    state = Dict{String,Any}(
        "stage" => "short", "iteration" => 1, "fit_months" => 3,
        "param_names" => names, "best_candidate_id" => "1", "best_score" => 1.0,
        "completed_count" => 1, "failed_count" => 0, "skipped_count" => 0, "pending_count" => 0,
        "historical_trajectory" => trajectory, "trajectory_identity" => trajectory["identity"],
        "prefix_hash" => prefix_hash, "locked_intervals" => locks, "cma_state" => cma,
        "archive_ids" => ["1"], "transfer_archive_ids" => ["1"])
    reusable = Dict{String,Any}(
        "stage" => "short", "param_names" => names, "mean" => values, "sigma" => [0.1, 0.1],
        "covariance" => [1.0 0.0; 0.0 1.0], "p_c" => [0.0, 0.0], "p_sigma" => [0.0, 0.0],
        "historical_trajectory" => trajectory, "trajectory_identity" => trajectory["identity"],
        "prefix_hash" => prefix_hash, "locked_intervals" => locks, "cma_state" => cma,
        "archive_ids" => ["1"], "selected_archive_ids" => ["1"],
        "archive_lineage" => Dict("canonical_archive_path" => abspath(joinpath(stage_root, "survivor_archive.json")),
                                  "archive_ids" => ["1"]))
    O.safe_save_json(joinpath(stage_root, "stage_state.json"), state)
    O.safe_save_json(joinpath(stage_root, "full_reusable_state.json"), reusable)
    O.safe_save_json(joinpath(stage_root, "survivor_archive_summary.json"), Dict("count" => 1))
    production_keys = ["stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
        "survivor_archive.json", "survivor_archive_summary.json", "full_reusable_state.json",
        "archive_transfer_manifest.json", joinpath("iter_1", "candidate_list.txt"),
        joinpath("iter_1", "cma_sampling_state.json"), joinpath("iter_1", "top_candidates.json")]
    hashes = Dict(k => bytes2hex(SHA.sha256(read(joinpath(stage_root, k)))) for k in production_keys)
    commit = Dict{String,Any}("status" => "committed", "schema_version" => "production-v1",
        "stage" => "short", "iteration" => 1, "artifact_key_set" => production_keys,
        "artifact_hashes" => hashes,
        "artifact_hash_manifest" => bytes2hex(SHA.sha256(JSON.json(hashes))),
        "artifact_integrity_digest" => bytes2hex(SHA.sha256(JSON.json(hashes))))
    O.safe_save_json(joinpath(iter_root, "iteration_commit.json"), commit)
    return stage_root
end

@testset "direct production predecessor loader context" begin
    root = mktempdir()
    cfg = production_config(root)
    stage_root = write_production_predecessor(cfg.output_dir)
    @test O.validate_committed_artifacts(stage_root, "production-v1")["valid"]
    loaded = O.load_immediate_predecessor_state(cfg, cfg.stages[2])
    @test loaded["stage_state"]["trajectory_identity"] == "trajectory-production-fixture"
    @test loaded["reusable_state"]["archive_ids"] == ["1"]
    @test loaded["archive"][1]["candidate"] == "1"

    commit_path = joinpath(stage_root, "iter_1", "iteration_commit.json")
    commit = O.load_json(commit_path)
    commit["schema_version"] = "fixture-v1"
    O.safe_save_json(commit_path, commit)
    @test !O.validate_committed_artifacts(stage_root, "production-v1")["valid"]
    @test_throws ArgumentError O.load_immediate_predecessor_state(cfg, cfg.stages[2])

    write_production_predecessor(cfg.output_dir)
    state_path = joinpath(stage_root, "stage_state.json")
    state = O.load_json(state_path)
    delete!(state, "historical_trajectory")
    O.safe_save_json(state_path, state)
    @test_throws ArgumentError O.load_immediate_predecessor_state(cfg, cfg.stages[2])

    # Fixture-shaped validation remains explicit and does not authorize production loading.
    @test !O.validate_committed_artifacts(stage_root, "fixture-v1")["valid"]
end
