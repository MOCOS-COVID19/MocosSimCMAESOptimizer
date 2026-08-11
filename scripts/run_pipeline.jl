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

function fixture_file_identity(path)
    p = path === nothing ? nothing : String(path)
    if p === nothing || isempty(p) || !isfile(p)
        return Dict{String,Any}("path" => p, "exists" => false)
    end
    return Dict{String,Any}("path" => abspath(p), "exists" => true,
        "sha256" => bytes2hex(sha256(read(p))))
end

function fixture_provenance(preflight, base_config, stage, index, stage_root)
    config_path = get(preflight, "config_path", nothing)
    seed_path = get(preflight, "seed_config", get(base_config, "seed_config", nothing))
    return Dict{String,Any}(
        "source_config" => fixture_file_identity(config_path),
        "source_seed" => fixture_file_identity(seed_path),
        "source_config_path" => config_path,
        "source_seed_path" => seed_path,
        "stage" => stage["name"], "iteration" => 1,
        "horizon" => Dict("requested_months" => stage["fit_months"],
            "effective_months" => stage["effective_months"],
            "requested_days" => stage["requested_days"],
            "effective_days" => stage["effective_days"]),
        "adapter_mode" => "deterministic_fixture",
        "adapter_command_identity" => "fixture-adapter-v1:no-launch",
        "metric_version" => "objective-metrics-v1",
        "output_paths" => Dict("stage_root" => stage_root,
            "metrics" => joinpath(stage_root, "metrics.json"),
            "archive" => joinpath(stage_root, "survivor_archive.json"),
            "selection_report" => joinpath(stage_root, "survivor_selection_report.json")),
        "pipeline_stage_index" => index)
end

function validate_fixture_provenance(stage_root, entries, archive, report, state, reusable)
    contradictions = String[]
    ids = Set(String(get(x, "candidate", "")) for x in archive)
    entry_ids = Set(String(get(x, "candidate", "")) for x in entries)
    ids ⊆ entry_ids || push!(contradictions, "archive candidate is absent from metrics")
    Int(get(report, "archive_count", -1)) == length(archive) ||
        push!(contradictions, "selection report archive_count disagrees with archive")
    if Int(get(report, "rejected_total", -1)) !=
       sum(values(get(report, "rejected_counts", Dict())))
        push!(contradictions, "selection report rejected_total disagrees with rejected_counts")
    end
    get(state, "archive_ids", Any[]) == [get(x, "candidate", nothing) for x in archive] ||
        push!(contradictions, "stage state archive_ids disagree with archive order")
    get(reusable, "admitted_ids", Any[]) == get(state, "transfer_archive_ids", Any[]) ||
        push!(contradictions, "reusable state admitted_ids disagree with transfer ids")
    get(reusable, "selected_archive_ids", Any[]) == [get(x, "candidate", nothing) for x in archive] ||
        push!(contradictions, "reusable state selected_archive_ids disagree with archive")
    for entry in entries
        for field in ("provenance", "score_evidence", "output_paths", "horizon",
                      "source_config_identity", "source_seed_identity")
            haskey(entry, field) || push!(contradictions, "candidate missing $field")
        end
    end
    return Dict{String,Any}("status" => isempty(contradictions) ? "consistent" : "contradictory",
        "contradictions" => contradictions, "candidate_count" => length(entries),
        "archive_count" => length(archive), "reusable_admitted_ids" => get(reusable, "admitted_ids", Any[]))
end

function validate_consumed_trusted_state(stage_root::String,
                                         expected_stage::String,
                                         expected_horizon::Int;
                                         previous_root::Union{Nothing,String}=nothing)
    state_path = joinpath(stage_root, "stage_state.json")
    reusable_path = joinpath(stage_root, "full_reusable_state.json")
    isfile(state_path) || error("missing trusted predecessor stage_state.json: $stage_root")
    isfile(reusable_path) || error("missing trusted predecessor full_reusable_state.json: $stage_root")
    state = JSON.parsefile(state_path)
    reusable = JSON.parsefile(reusable_path)
    state isa AbstractDict && reusable isa AbstractDict ||
        error("trusted predecessor state artifacts are malformed: $stage_root")
    String(get(state, "status", "")) == "committed" ||
        error("trusted predecessor stage is not committed: $expected_stage")
    String(get(reusable, "status", "")) == "committed" ||
        error("trusted predecessor reusable state is not committed: $expected_stage")
    String(get(state, "stage", "")) == expected_stage ||
        error("trusted predecessor stage identity mismatch: $expected_stage")
    String(get(reusable, "stage", "")) == expected_stage ||
        error("trusted predecessor reusable stage identity mismatch: $expected_stage")
    Int(get(state, "fit_months", -1)) == expected_horizon ||
        error("trusted predecessor horizon mismatch: $expected_stage")
    String(get(state, "trajectory_identity", "")) != "" ||
        error("trusted predecessor trajectory identity missing: $expected_stage")
    String(get(reusable, "trajectory_identity", "")) == String(state["trajectory_identity"]) ||
        error("trusted predecessor trajectory identity mismatch: $expected_stage")
    historical = get(state, "historical_trajectory", nothing)
    reusable_historical = get(reusable, "historical_trajectory", nothing)
    historical isa AbstractDict && reusable_historical isa AbstractDict ||
        error("trusted predecessor historical trajectory missing: $expected_stage")
    get(historical, "values", Any[]) == get(reusable_historical, "values", Any[]) ||
        error("trusted predecessor historical values mismatch: $expected_stage")
    String(get(state, "prefix_hash", "")) ==
        bytes2hex(sha256(JSON.json(get(historical, "prefix_values", Any[])))) ||
        error("trusted predecessor prefix hash mismatch: $expected_stage")
    String(get(reusable, "prefix_hash", "")) == String(state["prefix_hash"]) ||
        error("trusted predecessor reusable prefix hash mismatch: $expected_stage")
    locked = get(state, "locked_intervals", nothing)
    locked isa AbstractVector || error("trusted predecessor locked prefix missing: $expected_stage")
    cma = get(reusable, "cma_state", nothing)
    cma isa AbstractDict || error("trusted predecessor CMA state missing: $expected_stage")
    all(haskey(cma, field) for field in
        ("parameter_names", "mean", "sigma", "covariance", "p_c", "p_sigma")) ||
        error("trusted predecessor CMA state is incomplete: $expected_stage")
    state_cma = get(state, "cma_state", nothing)
    state_cma isa AbstractDict || error("trusted predecessor stage CMA state missing: $expected_stage")
    all(get(state_cma, field, nothing) == get(cma, field, nothing)
        for field in ("parameter_names", "mean", "sigma", "covariance", "p_c", "p_sigma")) ||
        error("trusted predecessor CMA state mismatch: $expected_stage")
    archive_path = abspath(joinpath(stage_root, "survivor_archive.json"))
    String(get(state, "archive_path", "")) == archive_path ||
        error("trusted predecessor archive path mismatch: $expected_stage")
    archive = JSON.parsefile(archive_path)
    archive_ids = [String(get(x, "candidate", "")) for x in archive]
    get(state, "archive_ids", Any[]) == archive_ids ||
        error("trusted predecessor archive ordering mismatch: $expected_stage")
    get(reusable, "selected_archive_ids", Any[]) == archive_ids ||
        error("trusted predecessor reusable archive ordering mismatch: $expected_stage")
    if previous_root !== nothing
        source_path = abspath(joinpath(previous_root, "survivor_archive.json"))
        String(get(state, "source_archive_path", "")) == source_path ||
            error("trusted predecessor source archive path mismatch: $expected_stage")
    end
    return state, reusable
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
    # The fixture models one trusted historical trajectory.  Its identity is
    # stable across stages, while each stage appends only its newly exposed
    # suffix.  This is deliberately deterministic and does not invoke a
    # simulator.
    trajectory_identity = bytes2hex(sha256("fixture-trusted-trajectory-v1"))
    historical_values = Float64[]
    previous_prefix_hash = bytes2hex(sha256(JSON.json(historical_values)))
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
            source_root = joinpath(output_root, source_stage)
            predecessor_state, predecessor_reusable =
                validate_consumed_trusted_state(
                    source_root, source_stage, Int(plan[index - 1]["fit_months"]);
                    previous_root=index > 2 ? joinpath(output_root, String(plan[index - 2]["name"])) : nothing)
            expected_path = joinpath(output_root, source_stage,
                                     "archive_transfer_manifest.json")
            expected_archive_path = joinpath(output_root, source_stage,
                                             "survivor_archive.json")
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
            incoming_path = expected_archive_path
            incoming_manifest = JSON.parsefile(expected_path)
            incoming_manifest["canonical_archive_path"] ==
                abspath(joinpath(source_root, "survivor_archive.json")) ||
                error("fixture transfer manifest archive path is not canonical")
            incoming_manifest["admitted_order"] == incoming_manifest["admitted_ids"] ||
                error("fixture transfer manifest ordering mismatch")
            incoming_manifest["admitted_ids"] == predecessor_state["archive_ids"] ||
                error("fixture transfer manifest does not consume predecessor state")
            incoming_manifest["admitted_ids"] == predecessor_reusable["selected_archive_ids"] ||
                error("fixture transfer manifest does not consume predecessor reusable archive")
        end
        stage_root = joinpath(output_root, String(stage["name"]))
        mkpath(stage_root)
        # The predecessor is the sole source of trusted history.  Rebuilding
        # this from fixture constants would let a mutated predecessor be
        # silently bypassed during extension.
        if index > 1
            historical_values = copy(predecessor_state["historical_trajectory"]["values"])
        end
        prefix_values = copy(historical_values)
        prefix_hash = bytes2hex(sha256(JSON.json(prefix_values)))
        requested_days = Int(stage["requested_days"])
        append!(historical_values,
            [0.2 + 0.001 * day for day in (length(historical_values) + 1):requested_days])
        locked_intervals = [
            Dict("name" => "fixture.trajectory[$day]", "start_day" => day,
                 "end_day" => day, "value" => value, "class" => "locked")
            for (day, value) in enumerate(prefix_values)
        ]
        historical_trajectory = Dict(
            "identity" => trajectory_identity,
            "prefix_values" => prefix_values,
            "prefix_hash" => prefix_hash,
            "values" => copy(historical_values),
            "start_day" => isempty(prefix_values) ? nothing : 1,
            "end_day" => requested_days,
            "initialized_suffix" => requested_days - length(prefix_values),
        )
        manifest = deepcopy(preflight)
        manifest["stage"] = stage
        manifest["source_stage"] = index == 1 ? nothing : plan[index - 1]["name"]
        manifest["output_root_created"] = true
        manifest["adapter_mode"] = "deterministic_fixture"
        stage_provenance = fixture_provenance(preflight, base_config, stage, index, stage_root)
        manifest["provenance"] = stage_provenance
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
                "score_evidence" => Dict("total" => 1.0 + 0.01 * slot,
                    "metrics" => Dict("weekly_control_score" => 0.8 + 0.01 * slot,
                        "daily_detections_cumulative" => 0.9,
                        "temporal_jump_penalty" => 0.0),
                    "metric_version" => "objective-metrics-v1", "scored_days" => stage["effective_days"]),
                "evaluated_vector" => [0.1 * slot, 0.2 * slot],
                "parameter_names" => ["fixture.beta[1]", "fixture.beta[2]"],
                "trajectory_identity" => trajectory_identity,
                "historical_trajectory" => deepcopy(historical_trajectory),
                "prefix_hash" => prefix_hash,
                "candidate_class" => index == 1 ? "new_dimension" : "new_dimension",
                "source_archive_id" => nothing,
                "admitted_predecessor_id" => nothing,
                "metrics" => Dict("weekly_control_score" => 0.8 + 0.01 * slot,
                                  "daily_detections_cumulative" => 0.9,
                                  "temporal_jump_penalty" => 0.0),
                "provenance" => Dict("adapter" => "deterministic_fixture",
                                     "source_config" => preflight["config_path"],
                                     "source_config_identity" => stage_provenance["source_config"],
                                     "source_seed_identity" => stage_provenance["source_seed"],
                                     "source_stage" => index == 1 ? nothing : plan[index - 1]["name"],
                                     "output_root" => stage_root,
                                     "adapter_mode" => "deterministic_fixture",
                                     "adapter_command_identity" => "fixture-adapter-v1:no-launch",
                                     "metric_version" => "objective-metrics-v1"),
                "source_config_identity" => stage_provenance["source_config"],
                "source_seed_identity" => stage_provenance["source_seed"],
                "horizon" => stage_provenance["horizon"],
                "output_paths" => Dict("stage_root" => stage_root,
                    "candidate_metrics" => joinpath(stage_root, "metrics.json"),
                    "candidate_output" => joinpath(stage_root, "candidate_outputs", id)),
                "adapter_mode" => "deterministic_fixture",
                "metric_version" => "objective-metrics-v1",
                "failure_class" => nothing,
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
        report["rejected_total"] = sum(values(report["rejected_counts"]))
        report["effective_quality_band"] = report["quality_band"]
        report["selection_stage"] = stage["name"]
        report["selection_iteration"] = 1
        report["source_config_identity"] = stage_provenance["source_config"]
        report["source_seed_identity"] = stage_provenance["source_seed"]
        report["horizon"] = stage_provenance["horizon"]
        report["adapter_mode"] = "deterministic_fixture"
        report["metric_version"] = "objective-metrics-v1"
        # Terminal classifications are persisted independently of the
        # selection result.  Resume must consume these records, rather than
        # inferring completion from candidate directories.
        O.safe_save_json(joinpath(stage_root, "top_candidates.json"), entries;
                         label="fixture_top_candidates")
        O.safe_save_json(joinpath(stage_root, "metrics.json"), entries;
                         label="fixture_candidate_metrics")
        open(joinpath(stage_root, "iter_metrics.jsonl"), "w") do io
            for entry in entries
                JSON.print(io, entry)
                print(io, '\n')
            end
        end
        archive_path = joinpath(stage_root, "survivor_archive.json")
        O.safe_save_json(archive_path, archive; label="fixture_survivor_archive")
        selection_report_path = joinpath(stage_root, "survivor_selection_report.json")
        O.safe_save_json(selection_report_path, report; label="fixture_survivor_selection_report")
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
        # Transfer records are immutable lineage records.  Add target-stage
        # classification without changing the canonical predecessor archive.
        transfer_archive = [
            merge(deepcopy(entry), Dict{String,Any}(
                "candidate_class" => "archive_transfer",
                "trajectory_identity" => trajectory_identity,
                "historical_trajectory" => deepcopy(historical_trajectory),
                "prefix_hash" => prefix_hash,
                "admitted_predecessor_id" => get(entry, "candidate", nothing),
                "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
                "locked_intervals" => deepcopy(locked_intervals),
            )) for entry in incoming_archive]
        O.safe_save_json(joinpath(stage_root, "transfer_candidates.json"),
                         transfer_archive; label="fixture_transfer_candidates")
        transfer = Dict("source_archive_path" => archive_path,
            "source_stage" => index == 1 ? nothing : plan[index - 1]["name"],
            "target_stage" => stage["name"],
            "source_horizon_months" => index == 1 ? nothing : plan[index - 1]["fit_months"],
            "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
            "admitted_ids" => transfer_ids, "candidate_order" => transfer_ids,
            "protected_transfer_slots" => transfer_ids, "immigrant_slots" => String[],
            "archive_lineage" => index == 1 ? nothing :
                joinpath(output_root, String(plan[index - 1]["name"]),
                         "archive_transfer_manifest.json"),
            "trajectory_identity" => trajectory_identity,
            "prefix_hash" => prefix_hash,
            "locked_intervals" => locked_intervals,
            "source_reusable_state_path" => index == 1 ? nothing :
                joinpath(output_root, String(plan[index - 1]["name"]), "full_reusable_state.json"))
        # For the first stage there is no predecessor; for later stages the
        # source path must remain the exact canonical predecessor manifest.
        index > 1 && (transfer["source_archive_path"] = incoming_path)
        O.safe_save_json(joinpath(stage_root, "transfer_manifest.json"), transfer;
                         label="fixture_transfer_manifest")
        O.safe_save_json(joinpath(stage_root, "stage_state.json"), Dict(
            "status" => "committed", "stage" => stage["name"], "iteration" => 1,
            "fit_months" => stage["fit_months"], "requested_days" => stage["requested_days"],
            "effective_days" => stage["effective_days"], "archive_path" => archive_path,
            "archive_manifest_path" => manifest_path,
            "archive_ids" => current_ids, "transfer_archive_ids" => transfer_ids,
            "selection_report_path" => selection_report_path,
            "selection_report_archive_ids" => current_ids,
            "provenance_validation_path" => joinpath(stage_root, "provenance_validation.json"),
            "best_candidate" => current_ids[1],
            "current_archive_ids" => current_ids,
            "source_archive_path" => incoming_path,
            "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
            "rng_state" => Dict("algorithm" => "fixture-fixed", "stream" => index),
            "trajectory_identity" => trajectory_identity,
            "historical_trajectory" => historical_trajectory,
            "prefix_hash" => prefix_hash,
            "locked_intervals" => locked_intervals,
            "new_suffix_days" => requested_days - length(prefix_values),
            "previous_prefix_hash" => previous_prefix_hash))
        state = JSON.parsefile(joinpath(stage_root, "stage_state.json"))
        cma_state = Dict{String,Any}(
            "state_id" => string(stage["name"], ":cma:", trajectory_identity),
            "parameter_names" => ["fixture.beta[1]", "fixture.beta[2]"],
            "mean" => [0.1, 0.2], "sigma" => [0.08, 0.08],
            "covariance" => [[0.04, 0.0], [0.0, 0.04]],
            "p_c" => [0.0, 0.0], "p_sigma" => [0.0, 0.0],
            "source_archive_ids" => transfer_ids)
        state["cma_state"] = cma_state
        O.safe_save_json(joinpath(stage_root, "stage_state.json"), state;
                         label="fixture_stage_state_with_cma")
        reusable_state = Dict(
            "status" => "committed", "stage" => stage["name"],
            "source_archive_path" => incoming_path, "admitted_ids" => transfer_ids,
            "parameter_names" => ["fixture.beta[1]", "fixture.beta[2]"],
            "transition_delta_report" => [x["transition_delta_report"] for x in archive],
            "trajectory_identity" => trajectory_identity,
            "prefix_hash" => prefix_hash,
            "historical_trajectory" => historical_trajectory,
            "locked_intervals" => locked_intervals,
            "cma_state" => cma_state,
            "source_archive_ids" => transfer_ids,
            "selected_archive_ids" => current_ids,
            "state_provenance" => Dict("source" => incoming_path,
                "admitted_predecessor_ids" => transfer_ids,
                "scalar_best_usable" => false),
            "selection_report_path" => selection_report_path,
            "selection_report_archive_ids" => current_ids,
            "provenance" => stage_provenance)
        O.safe_save_json(joinpath(stage_root, "full_reusable_state.json"), reusable_state)
        state = JSON.parsefile(joinpath(stage_root, "stage_state.json"))
        provenance_check = validate_fixture_provenance(stage_root, entries, archive, report, state, reusable_state)
        O.safe_save_json(joinpath(stage_root, "provenance_validation.json"), provenance_check;
                         label="fixture_provenance_validation")
        provenance_check["status"] == "consistent" ||
            error("fixture provenance contradiction: $(provenance_check["contradictions"])")
        committed_files = ["preflight_manifest.json", "stage_state.json",
            "iter_metrics.jsonl", "top_candidates.json", "survivor_archive.json",
            "survivor_selection_report.json", "full_reusable_state.json",
            "transfer_manifest.json", "transfer_candidates.json",
            "provenance_validation.json", "archive_transfer_manifest.json"]
        artifact_hashes = Dict{String,Any}(
            file => bytes2hex(sha256(read(joinpath(stage_root, file))))
            for file in committed_files)
        O.atomic_save_json(joinpath(stage_root, "iter_1", "iteration_commit.json"),
            let commit = Dict("status" => "committed", "stage" => stage["name"],
                              "iteration" => 1,
                              "schema_version" => "fixture-v1",
                              "candidate_ids" => [String(x["candidate"]) for x in entries],
                              "artifact_hashes" => artifact_hashes,
                              "artifact_key_set" => committed_files)
                commit["artifact_hash_manifest"] = fixture_hash_manifest_digest(artifact_hashes)
                commit
            end)
        push!(stages, merge(stage, Dict("stage_root" => stage_root,
            "archive_path" => archive_path, "archive_manifest_path" => manifest_path,
            "archive_ids" => current_ids,
            "transfer_archive_ids" => transfer_ids,
            "current_archive_ids" => current_ids,
            "selection_report_path" => selection_report_path,
            "provenance_validation_path" => joinpath(stage_root, "provenance_validation.json"),
            "source_archive_path" => incoming_path,
            "source_archive_id" => incoming_manifest === nothing ? nothing : incoming_manifest["archive_id"],
            "trajectory_identity" => trajectory_identity,
            "prefix_hash" => prefix_hash,
            "gate" => gate,
            "status" => "committed")))
        previous_prefix_hash = prefix_hash
    end
    summary = Dict("status" => "fixture_complete", "adapter_mode" => "deterministic_fixture",
        "output_root" => output_root, "target_months" => target, "monthly_days" => monthly_days,
        "stages" => stages, "deferred" => Dict("simulation" => "DEFERRED",
            "multi_seed" => "DEFERRED", "slurm" => "DEFERRED",
            "validation_replicates" => "DEFERRED"),
        "provenance" => Dict("config_path" => preflight["config_path"],
            "threshold" => 0.9, "preflight_before_execution" => true,
            "metric_version" => "objective-metrics-v1",
            "selection_reports" => [s["selection_report_path"] for s in stages]))
    O.safe_save_json(joinpath(output_root, "pipeline_summary.json"), summary;
                     label="fixture_pipeline_summary")
    return summary
end

function fixture_artifact_hash(path)
    return bytes2hex(sha256(read(path)))
end

function fixture_hash_manifest_digest(hashes)
    ordered = Dict{String,Any}(
        String(k) => hashes[k] for k in sort(String.(collect(keys(hashes))))
    )
    return bytes2hex(sha256(JSON.json(ordered)))
end

function validate_fixture_stage_root(root::String, previous_root::Union{Nothing,String}=nothing)
    required = ["preflight_manifest.json", "stage_state.json", "iter_metrics.jsonl",
        "top_candidates.json", "survivor_archive.json", "survivor_selection_report.json",
        "full_reusable_state.json", "transfer_manifest.json", "transfer_candidates.json",
        "provenance_validation.json", "archive_transfer_manifest.json",
        "iter_1/iteration_commit.json"]
    all(isfile(joinpath(root, f)) for f in required) ||
        error("fixture stage is missing committed artifacts: $(basename(root))")
    state = JSON.parsefile(joinpath(root, "stage_state.json"))
    reusable = JSON.parsefile(joinpath(root, "full_reusable_state.json"))
    archive = JSON.parsefile(joinpath(root, "survivor_archive.json"))
    transfer = JSON.parsefile(joinpath(root, "transfer_manifest.json"))
    selection = JSON.parsefile(joinpath(root, "survivor_selection_report.json"))
    top = JSON.parsefile(joinpath(root, "top_candidates.json"))
    metrics = Any[]
    for line in eachline(joinpath(root, "iter_metrics.jsonl"))
        isempty(strip(line)) || push!(metrics, JSON.parse(line))
    end
    commit = JSON.parsefile(joinpath(root, "iter_1", "iteration_commit.json"))
    state isa AbstractDict && reusable isa AbstractDict && archive isa AbstractVector &&
        top isa AbstractVector && transfer isa AbstractDict ||
        error("fixture stage artifacts are malformed: $(basename(root))")
    String(get(commit, "schema_version", "")) == "fixture-v1" ||
        error("fixture iteration commit schema is missing or downgraded: $stage")
    get(state, "status", "") == "committed" || error("fixture stage state is not committed")
    get(reusable, "status", "") == "committed" || error("fixture reusable state is not committed")
    stage = String(get(state, "stage", ""))
    !isempty(stage) && String(get(reusable, "stage", "")) == stage ||
        error("fixture trusted-state stage identity mismatch: $stage")
    String(get(state, "archive_path", "")) == abspath(joinpath(root, "survivor_archive.json")) ||
        error("fixture stage archive path is not canonical: $stage")
    String(get(reusable, "trajectory_identity", "")) == String(get(state, "trajectory_identity", "")) ||
        error("fixture trajectory identity mismatch: $stage")
    String(get(reusable, "prefix_hash", "")) == String(get(state, "prefix_hash", "")) ||
        error("fixture trusted prefix hash mismatch: $stage")
    haskey(reusable, "cma_state") || error("fixture reusable state lacks CMA state: $stage")
    cma = reusable["cma_state"]
    for field in ("parameter_names", "mean", "sigma", "covariance", "p_c", "p_sigma")
        haskey(cma, field) || error("fixture CMA state lacks $field: $stage")
    end
    state_cma = get(state, "cma_state", nothing)
    state_cma isa AbstractDict &&
        all(get(state_cma, field, nothing) == get(cma, field, nothing)
            for field in ("parameter_names", "mean", "sigma", "covariance", "p_c", "p_sigma")) ||
        error("fixture stage/reusable CMA state mismatch: $stage")
    archive_ids = [String(get(x, "candidate", "")) for x in archive]
    get(state, "archive_ids", Any[]) == archive_ids ||
        error("fixture state/archive IDs disagree: $stage")
    get(reusable, "selected_archive_ids", Any[]) == archive_ids ||
        error("fixture reusable/archive IDs disagree: $stage")
    length(top) == length(metrics) ||
        error("fixture top candidates/metrics length mismatch: $stage")
    for row in metrics
        row isa AbstractDict && get(row, "status", "") in ("completed", "failed", "skipped") ||
            error("fixture metrics contain nonterminal candidate: $stage")
        all(haskey(row, field) for field in
            ("score_evidence", "output_paths", "horizon", "source_config_identity",
             "source_seed_identity", "adapter_mode", "metric_version", "failure_class")) ||
            error("fixture candidate provenance is incomplete: $stage")
    end
    commit["status"] == "committed" && String(commit["stage"]) == stage ||
        error("fixture commit identity mismatch: $stage")
    hashes = get(commit, "artifact_hashes", nothing)
    hashes isa AbstractDict || error("fixture commit has no artifact hashes: $stage")
    expected_keys = sort(String.(required[1:end-1]))
    sort(String.(collect(keys(hashes)))) == expected_keys ||
        error("fixture artifact hash manifest has an inexact key set: $stage")
    get(commit, "artifact_key_set", Any[]) == required[1:end-1] ||
        error("fixture artifact key set is not exact: $stage")
    String(get(commit, "artifact_hash_manifest", "")) ==
        fixture_hash_manifest_digest(hashes) ||
        error("fixture artifact hash manifest integrity mismatch: $stage")
    for (file, expected) in hashes
        path = joinpath(root, String(file))
        isfile(path) && fixture_artifact_hash(path) == String(expected) ||
            error("fixture artifact content hash mismatch: $stage/$file")
    end
    if previous_root !== nothing
        previous_archive = abspath(joinpath(previous_root, "survivor_archive.json"))
        String(get(transfer, "source_archive_path", "")) == previous_archive ||
            error("fixture transfer does not name canonical predecessor archive: $stage")
        String(get(transfer, "archive_lineage", "")) ==
            abspath(joinpath(previous_root, "archive_transfer_manifest.json")) ||
            error("fixture transfer manifest lineage mismatch: $stage")
        get(transfer, "admitted_ids", Any[]) == get(transfer, "candidate_order", Any[]) ||
            error("fixture transfer ordering mismatch: $stage")
        evidence = O.load_transfer_survivor_archive(dirname(root), stage;
            predecessor_stage=basename(previous_root),
            expected_fit_months=Int(JSON.parsefile(joinpath(previous_root,
                "stage_state.json"))["fit_months"]),
            expected_manifest_path=joinpath(previous_root, "archive_transfer_manifest.json"),
            stage_order=sort!(String.(basename.(filter(isdir, readdir(dirname(root), join=true))))),
            return_evidence=true)
        evidence isa AbstractVector ||
            error("fixture transfer archive validation failed: $stage")
    else
        get(transfer, "source_stage", nothing) === nothing ||
            error("fixture first stage unexpectedly has a predecessor")
    end
    return (state=state, reusable=reusable, archive=archive, transfer=transfer,
            selection=selection, commit=commit)
end

function reconstruct_fixture_summary(output_root::String)
    dirs = sort([joinpath(output_root, name) for name in readdir(output_root)
                 if isdir(joinpath(output_root, name))])
    isempty(dirs) && error("fixture output root has no committed stages")
    stages = Any[]
    previous = nothing
    for root in dirs
        checked = validate_fixture_stage_root(root, previous)
        state, archive, transfer, selection = checked.state, checked.archive,
            checked.transfer, checked.selection
        fit = Int(state["fit_months"])
        push!(stages, Dict{String,Any}(
            "name" => String(state["stage"]), "fit_months" => fit,
            "requested_days" => Int(state["requested_days"]),
            "effective_months" => fit, "effective_days" => Int(state["effective_days"]),
            "remaining_months" => 0, "stage_root" => root,
            "archive_path" => abspath(joinpath(root, "survivor_archive.json")),
            "archive_manifest_path" => abspath(joinpath(root, "archive_transfer_manifest.json")),
            "archive_ids" => [String(get(x, "candidate", "")) for x in archive],
            "transfer_archive_ids" => get(state, "transfer_archive_ids", Any[]),
            "current_archive_ids" => [String(get(x, "candidate", "")) for x in archive],
            "selection_report_path" => joinpath(root, "survivor_selection_report.json"),
            "provenance_validation_path" => joinpath(root, "provenance_validation.json"),
            "source_archive_path" => get(state, "source_archive_path", nothing),
            "source_archive_id" => get(state, "source_archive_id", nothing),
            "trajectory_identity" => state["trajectory_identity"],
            "prefix_hash" => state["prefix_hash"],
            "gate" => JSON.parsefile(joinpath(root, "stage_extension_gate.json")),
            "status" => "committed"))
        previous = root
    end
    first_manifest = JSON.parsefile(joinpath(first(dirs), "preflight_manifest.json"))
    target = maximum(Int(s["fit_months"]) for s in stages)
    for stage in stages
        stage["remaining_months"] = target - Int(stage["fit_months"])
    end
    summary = Dict{String,Any}("status" => "fixture_complete",
        "adapter_mode" => "deterministic_fixture", "output_root" => abspath(output_root),
        "target_months" => target, "monthly_days" => Int(get(first_manifest, "monthly_days", 30)),
        "stages" => stages,
        "deferred" => Dict("simulation" => "DEFERRED", "multi_seed" => "DEFERRED",
            "slurm" => "DEFERRED", "validation_replicates" => "DEFERRED"),
        "provenance" => Dict("config_path" => first_manifest["config_path"],
            "threshold" => 0.9, "preflight_before_execution" => true,
            "metric_version" => "objective-metrics-v1",
            "selection_reports" => [s["selection_report_path"] for s in stages]))
    return summary
end

"""Load a previously completed fixture root without replaying its stages.

The fixture adapter has no external work to poll, so a committed summary is
the only safe resume point.  Validate the cross-stage joins before returning
it; in particular, do not let directory existence stand in for terminal
state.
"""
function resume_fixture_pipeline(output_root::String)
    summary_path = joinpath(output_root, "pipeline_summary.json")
    if !isfile(summary_path)
        summary = reconstruct_fixture_summary(output_root)
        O.safe_save_json(summary_path, summary; label="reconstructed_fixture_pipeline_summary")
        return summary
    end
    summary = try
        JSON.parsefile(summary_path)
    catch err
        error("fixture pipeline summary is malformed: $(sprint(showerror, err))")
    end
    status = String(get(summary, "status", ""))
    if status == "fixture_blocked"
        isfile(joinpath(output_root, "blocked_state.json")) ||
            error("blocked fixture root lacks durable blocked_state.json")
        return summary
    end
    status == "fixture_complete" ||
        error("fixture output root is not committed; refusing unsafe resume")
    stages = get(summary, "stages", nothing)
    stages isa AbstractVector && !isempty(stages) ||
        error("committed fixture summary has no stages")
    previous_name = nothing
    for stage in stages
        stage isa AbstractDict || error("committed fixture stage is malformed")
        name = String(get(stage, "name", ""))
        root = String(get(stage, "stage_root", joinpath(output_root, name)))
        root == joinpath(output_root, name) ||
            error("fixture stage root escapes output root: $name")
        get(stage, "status", "") == "committed" ||
            error("fixture stage is not committed: $name")
        previous_root = previous_name === nothing ? nothing :
            joinpath(output_root, previous_name)
        validate_fixture_stage_root(root, previous_root)
        required = ("preflight_manifest.json", "stage_state.json",
                    "iter_metrics.jsonl", "top_candidates.json",
                    "survivor_archive.json", "full_reusable_state.json",
                    "transfer_manifest.json", "transfer_candidates.json",
                    "iter_1/iteration_commit.json")
        all(isfile(joinpath(root, file)) for file in required) ||
            error("fixture stage is missing committed artifacts: $name")
        state = try JSON.parsefile(joinpath(root, "stage_state.json"))
        catch err
            error("fixture stage state is malformed: $name")
        end
        reusable = try JSON.parsefile(joinpath(root, "full_reusable_state.json"))
        catch err
            error("fixture reusable state is malformed: $name")
        end
        get(state, "status", "") == "committed" ||
            error("fixture stage state is not committed: $name")
        get(reusable, "status", "") == "committed" ||
            error("fixture reusable state is not committed: $name")
        String(get(state, "stage", "")) == name ||
            error("fixture stage state identity mismatch: $name")
        String(get(reusable, "stage", "")) == name ||
            error("fixture reusable state identity mismatch: $name")
        archive = try JSON.parsefile(joinpath(root, "survivor_archive.json"))
        catch err
            error("fixture archive is malformed: $name")
        end
        archive isa AbstractVector && !isempty(archive) ||
            error("fixture archive is empty or malformed: $name")
        terminal_rows = Any[]
        for line in eachline(joinpath(root, "iter_metrics.jsonl"))
            isempty(strip(line)) || push!(terminal_rows, JSON.parse(line))
        end
        terminal_rows isa AbstractVector && !isempty(terminal_rows) ||
            error("fixture terminal classifications are missing: $name")
        all(row isa AbstractDict &&
            get(row, "status", "") in ("completed", "failed", "skipped")
            for row in terminal_rows) ||
            error("fixture terminal classification is nonterminal: $name")
        top = JSON.parsefile(joinpath(root, "top_candidates.json"))
        top isa AbstractVector && length(top) == length(terminal_rows) ||
            error("fixture top candidates do not join terminal metrics: $name")
        commit = JSON.parsefile(joinpath(root, "iter_1", "iteration_commit.json"))
        get(commit, "status", "") == "committed" &&
            String(get(commit, "stage", "")) == name &&
            Int(get(commit, "iteration", 0)) == 1 ||
            error("fixture iteration commit is invalid: $name")
        admitted = get(reusable, "admitted_ids", Any[])
        transfer_rows = JSON.parsefile(joinpath(root, "transfer_candidates.json"))
        known_ids = Set{Any}(vcat(
            [get(entry, "candidate", nothing) for entry in archive],
            transfer_rows isa AbstractVector ?
                [get(entry, "candidate", nothing) for entry in transfer_rows] : Any[]))
        all(id in known_ids for id in admitted) ||
            error("fixture reusable state references an unarchived candidate: $name")
        if previous_name !== nothing
            transfer = JSON.parsefile(joinpath(root, "transfer_manifest.json"))
            String(get(transfer, "source_stage", "")) == previous_name ||
                error("fixture predecessor lineage mismatch: $name")
        end
        previous_name = name
    end
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
        output_root = resolve_path(String(batch["output_root"]), batch_dir)
        if ispath(output_root)
            return resume_fixture_pipeline(output_root)
        end
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
