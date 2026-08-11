#!/usr/bin/env julia

using JSON
using SHA

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function resolve_path(path::String, base::String)
    return isabspath(path) ? path : normpath(joinpath(base, path))
end

function phase_config(base_config, phase, output_dir, seed_config, batch_base)
    cfg = deepcopy(base_config)
    cfg["output_dir"] = output_dir
    cfg["seed_config"] = seed_config
    cfg["stages"] = [Dict(
        "name" => String(phase["name"]),
        "fit_months" => Int(phase["fit_months"]),
        "max_iterations" => Int(phase["max_iterations"]),
        "population_size" => Int(phase["population_size"]),
        "sigma" => Float64(phase["sigma"]),
    )]
    early_stop = get(batch_base, "early_stop", Dict{String,Any}())
    objective = deepcopy(cfg["objective"])
    objective["min_completion_fraction"] = get(
        early_stop, "min_completion_fraction",
        get(objective, "min_completion_fraction", 1.0),
    )
    objective["finish_iter_delay"] = get(
        early_stop, "finish_iter_delay",
        get(objective, "finish_iter_delay", 30),
    )
    cfg["objective"] = objective
    return cfg
end

function write_phase_config(path, cfg)
    open(path, "w") do io
        JSON.print(io, cfg, 2)
    end
end

function validate_fixture_stage_plan(batch, monthly_days)
    raw_stages = get(batch, "stages", nothing)
    raw_stages isa AbstractVector && !isempty(raw_stages) ||
        error("fixture stage plan must be a non-empty array")
    target = Int(get(batch, "target_months", 24))
    target > 0 || error("fixture target_months must be positive")
    stages = Any[]
    previous = 0
    for (i, raw) in enumerate(raw_stages)
        raw isa AbstractDict || error("stages[$i] must be an object")
        name = String(get(raw, "name", ""))
        months = Int(get(raw, "fit_months", 0))
        !isempty(name) && months > previous ||
            error("fixture stages[$i] must be strictly increasing and named")
        months <= target || error("fixture stages[$i] exceeds target_months")
        push!(stages, Dict{String,Any}(
            "name" => name, "fit_months" => months,
            "requested_days" => months * monthly_days,
            "effective_months" => months,
            "effective_days" => months * monthly_days,
            "remaining_months" => target - months,
        ))
        previous = months
    end
    previous == target || error("fixture stage plan is incomplete_target")
    return stages, target
end

function run_fixture_pipeline(batch_path::String, base_config::AbstractDict,
                              preflight::AbstractDict, batch::AbstractDict)
    monthly_days = Int(get(base_config, "monthly_days", 30))
    plan, target = validate_fixture_stage_plan(batch, monthly_days)
    batch_dir = dirname(abspath(batch_path))
    output_root = resolve_path(String(batch["output_root"]), batch_dir)
    ispath(output_root) && error("fixture output root already exists: $output_root")
    mkpath(output_root)
    stages = Any[]
    for (index, stage) in enumerate(plan)
        # Transfer is a read-only handoff from the one authoritative,
        # immediate predecessor.  Validate it before creating the target
        # stage root so stale, sibling, scalar-only, or tampered sources
        # fail closed without leaving a misleading next-stage artifact.
        incoming_archive = Any[]
        incoming_path = nothing
        incoming_manifest = nothing
        if index > 1
            source_stage = String(plan[index - 1]["name"])
            expected_path = joinpath(output_root, source_stage,
                                     "archive_transfer_manifest.json")
            evidence = O.load_transfer_survivor_archive(
                output_root, String(stage["name"]);
                predecessor_stage=source_stage,
                expected_fit_months=Int(plan[index - 1]["fit_months"]),
                expected_manifest_path=expected_path,
                stage_order=[String(x["name"]) for x in plan],
                return_evidence=true)
            evidence isa AbstractDict && get(evidence, "status", "") == "rejected" &&
                error("fixture predecessor archive rejected: $(evidence["failure_class"])")
            incoming_archive = O.load_transfer_survivor_archive(
                output_root, String(stage["name"]);
                predecessor_stage=source_stage,
                expected_fit_months=Int(plan[index - 1]["fit_months"]),
                expected_manifest_path=expected_path,
                stage_order=[String(x["name"]) for x in plan])
            incoming_path = expected_path
            incoming_manifest = JSON.parsefile(expected_path)
        end
        stage_root = joinpath(output_root, String(stage["name"]))
        mkpath(stage_root)
        manifest = deepcopy(preflight)
        manifest["stage"] = stage
        manifest["source_stage"] = index == 1 ? nothing : plan[index - 1]["name"]
        manifest["output_root_created"] = true
        manifest["adapter_mode"] = "deterministic_fixture"
        O.safe_save_json(joinpath(stage_root, "preflight_manifest.json"), manifest;
                         label="fixture_preflight_manifest")
        entries = Any[]
        for slot in 1:3
            id = "$(stage["name"])-survivor-$slot"
            push!(entries, Dict{String,Any}(
                "candidate" => id, "status" => "completed",
                "stage" => stage["name"], "iteration" => 1,
                "fit_months" => stage["fit_months"],
                "requested_horizon" => stage["fit_months"],
                "effective_scoring_horizon" => stage["fit_months"],
                "score" => 1.0 + 0.01 * slot,
                "evaluated_vector" => [0.1 * slot, 0.2 * slot],
                "parameter_names" => ["fixture.beta[1]", "fixture.beta[2]"],
                "metrics" => Dict("weekly_control_score" => 0.8 + 0.01 * slot,
                                  "daily_detections_cumulative" => 0.9,
                                  "temporal_jump_penalty" => 0.0),
                "provenance" => Dict("adapter" => "deterministic_fixture",
                                     "source_config" => preflight["config_path"],
                                     "source_stage" => index == 1 ? nothing : plan[index - 1]["name"],
                                     "output_root" => stage_root),
                "transition_delta_report" => Dict(
                    "coordinate_space" => "effective_named",
                    "candidate_class" => index == 1 ? "new_dimension" : "archive_transfer",
                    "max_abs_delta" => index == 1 ? 0.0 : 0.02,
                    "policy_outcome" => "accepted"),
            ))
        end
        # Current-stage survivor selection is intentionally independent from
        # the incoming transfer archive.  The latter is persisted verbatim as
        # transfer_candidates and never re-selected or replaced by immigrants.
        report = O.survivor_archive_update(Any[], entries;
            current_stage=stage["name"], current_fit_months=stage["fit_months"],
            target_size=3, max_size=200, return_report=true)
        archive = report["archive"]
        archive_path = joinpath(stage_root, "survivor_archive.json")
        O.safe_save_json(archive_path, archive; label="fixture_survivor_archive")
        manifest_path = O.persist_archive_transfer_manifest(stage_root, archive;
            archive_path=archive_path, stage=stage["name"], fit_months=stage["fit_months"])
        gate = O.archive_quality_gate(archive; current_stage=stage["name"],
            current_fit_months=stage["fit_months"], current_best_score=1.01,
            current_quality_band=Dict("threshold" => 1.10),
            current_minimum_size=1, current_diversity_passed=true, target_size=3)
        gate["status"] == "passed" || error("fixture stage gate blocked: $(stage["name"])")
        O.safe_save_json(joinpath(stage_root, "stage_extension_gate.json"), gate;
                         label="fixture_stage_gate")
        transfer_ids = incoming_manifest === nothing ? String[] :
            String.(incoming_manifest["admitted_order"])
        current_ids = [String(x["candidate"]) for x in archive]
        transfer_archive = incoming_archive
        O.safe_save_json(joinpath(stage_root, "transfer_candidates.json"),
                         transfer_archive; label="fixture_transfer_candidates")
        transfer = Dict("source_archive_path" => manifest_path,
            "source_stage" => index == 1 ? nothing : plan[index - 1]["name"],
            "target_stage" => stage["name"],
            "source_horizon_months" => index == 1 ? nothing : plan[index - 1]["fit_months"],
            "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
            "admitted_ids" => transfer_ids, "candidate_order" => transfer_ids,
            "protected_transfer_slots" => transfer_ids, "immigrant_slots" => String[],
            "archive_lineage" => incoming_path)
        # For the first stage there is no predecessor; for later stages the
        # source path must remain the exact canonical predecessor manifest.
        index > 1 && (transfer["source_archive_path"] = incoming_path)
        O.safe_save_json(joinpath(stage_root, "transfer_manifest.json"), transfer;
                         label="fixture_transfer_manifest")
        O.safe_save_json(joinpath(stage_root, "stage_state.json"), Dict(
            "status" => "committed", "stage" => stage["name"], "iteration" => 1,
            "fit_months" => stage["fit_months"], "requested_days" => stage["requested_days"],
            "effective_days" => stage["effective_days"], "archive_path" => manifest_path,
            "archive_ids" => current_ids, "transfer_archive_ids" => transfer_ids,
            "best_candidate" => current_ids[1],
            "current_archive_ids" => current_ids,
            "source_archive_path" => incoming_path,
            "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
            "rng_state" => Dict("algorithm" => "fixture-fixed", "stream" => index),
            "prefix_hash" => bytes2hex(sha256(JSON.json(transfer_archive)))))
        O.safe_save_json(joinpath(stage_root, "full_reusable_state.json"), Dict(
            "status" => "committed", "stage" => stage["name"],
            "source_archive_path" => incoming_path, "admitted_ids" => transfer_ids,
            "parameter_names" => ["fixture.beta[1]", "fixture.beta[2]"],
            "transition_delta_report" => [x["transition_delta_report"] for x in archive]))
        push!(stages, merge(stage, Dict("stage_root" => stage_root,
            "archive_path" => manifest_path, "archive_ids" => current_ids,
            "transfer_archive_ids" => transfer_ids,
            "current_archive_ids" => current_ids,
            "source_archive_path" => incoming_path,
            "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
            "gate" => gate,
            "status" => "committed")))
    end
    summary = Dict("status" => "fixture_complete", "adapter_mode" => "deterministic_fixture",
        "output_root" => output_root, "target_months" => target, "monthly_days" => monthly_days,
        "stages" => stages, "deferred" => Dict("simulation" => "DEFERRED",
            "multi_seed" => "DEFERRED", "slurm" => "DEFERRED",
            "validation_replicates" => "DEFERRED"),
        "provenance" => Dict("config_path" => preflight["config_path"],
            "threshold" => 0.9, "preflight_before_execution" => true))
    O.safe_save_json(joinpath(output_root, "pipeline_summary.json"), summary;
                     label="fixture_pipeline_summary")
    return summary
end

function run_pipeline(batch_path::String)
    batch_dir = dirname(abspath(batch_path))
    batch = JSON.parsefile(batch_path)
    base_path = resolve_path(String(batch["base_config"]), batch_dir)
    # Validate all source paths and schemas before touching the requested
    # output root. This is intentionally separate from optimizer execution.
    preflight_config(base_path; readiness=false)
    base_config = JSON.parsefile(base_path)
    if get(batch, "adapter_mode", "") == "fixture"
        return run_fixture_pipeline(batch_path, base_config,
                                     preflight_config(base_path; readiness=true), batch)
    end
    if haskey(base_config, "gt_dir")
        base_config["gt_dir"] = resolve_path(
            String(base_config["gt_dir"]),
            dirname(abspath(base_path)),
        )
    end
    output_root = resolve_path(String(batch["output_root"]), batch_dir)
    ispath(output_root) && error("pipeline output root already exists: $output_root")
    mkpath(output_root)
    use_slurm = Bool(get(batch, "use_slurm", true))

    short_phase = batch["short"]
    short_output = joinpath(output_root, String(short_phase["name"]))
    short_seed = resolve_path(
        String(get(short_phase, "seed_config", base_config["seed_config"])),
        batch_dir,
    )
    short_cfg = phase_config(base_config, short_phase, short_output, short_seed, batch)
    short_cfg_path = joinpath(output_root, "short_optimizer_config.json")
    write_phase_config(short_cfg_path, short_cfg)

    previous_posterior = get(batch, "initial_posterior", nothing)
    if previous_posterior !== nothing
        posterior_path = resolve_path(String(previous_posterior), batch_dir)
        reusable = posterior_reusable_state(posterior_path)
        safe_save_json(
            joinpath(short_output, "full_reusable_state.json"),
            reusable;
            label="initial_posterior_reusable_state",
        )
    end

    short_result = nothing
    if !Bool(get(batch, "skip_short", false))
        short_result = run_optimizer(short_cfg_path; use_slurm=use_slurm)
    elseif previous_posterior === nothing
        error("skip_short=true requires initial_posterior")
    end

    short_posterior = joinpath(
        short_output,
        "real_sims",
        String(short_phase["name"]),
        "posterior_samples.json",
    )
    isfile(short_posterior) || error("Short CMA-ES did not produce $short_posterior")
    short_best = joinpath(short_output, "final_best_candidate.json")
    isfile(short_best) || error("Short CMA-ES did not produce $short_best")

    long_phase = batch["long"]
    long_output = joinpath(output_root, String(long_phase["name"]))
    long_cfg_path = joinpath(output_root, "long_optimizer_config.json")
    long_seed = short_best
    long_cfg = phase_config(base_config, long_phase, long_output, long_seed, batch)
    write_phase_config(long_cfg_path, long_cfg)
    long_reusable = posterior_reusable_state(short_posterior)
    mkpath(long_output)
    safe_save_json(
        joinpath(long_output, "full_reusable_state.json"),
        long_reusable;
        label="short_posterior_reusable_state",
    )
    long_result = run_optimizer(long_cfg_path; use_slurm=use_slurm)

    summary = Dict(
        "batch_config" => abspath(batch_path),
        "short_config" => short_cfg_path,
        "short_output" => short_output,
        "short_posterior" => short_posterior,
        "long_config" => long_cfg_path,
        "long_output" => long_output,
        "long_posterior" => joinpath(long_output, "posterior_samples.json"),
        "short_result" => short_result,
        "long_result" => long_result,
    )
    safe_save_json(joinpath(output_root, "pipeline_summary.json"), summary; label="pipeline_summary")
    return summary
end

function run_readiness(config_path::String)
    manifest = preflight_config(config_path; readiness=true)
    parent = dirname(manifest["paths"]["output_dir"])
    root = create_candidate_root(parent, "readiness")
    manifest["readiness_root"] = root
    manifest["expected_artifacts"] = ["preflight_manifest.json", "no_simulation_invocations"]
    manifest["deferred"] = ["advanced_cli.jl", "validation_replicates", "Slurm"]
    safe_save_json(joinpath(root, "preflight_manifest.json"), manifest; label="readiness_manifest")
    return manifest
end

if length(ARGS) == 2 && ARGS[1] == "--readiness"
    println(JSON.json(run_readiness(ARGS[2])))
elseif length(ARGS) == 1
    println(JSON.json(run_pipeline(ARGS[1])))
else
    error("Usage: julia scripts/run_pipeline.jl [--readiness] pipeline.json")
end
