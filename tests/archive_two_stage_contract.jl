using Test
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function candidate(id, score, vector; stage="short", months=3, status="completed",
                   weekly=score, cumulative=score, age=score, penalty=score)
    Dict{String,Any}(
        "candidate" => id,
        "status" => status,
        "stage" => stage,
        "fit_months" => months,
        "requested_horizon" => months,
        "effective_scoring_horizon" => months,
        "score" => score,
        "evaluated_vector" => vector,
        "parameter_names" => ["beta[1]", "beta[2]"],
        "metrics" => Dict(
            "weekly_control_score" => weekly,
            "daily_detections_cumulative" => cumulative,
            "daily_age_05_14_detections" => age,
            "temporal_jump_penalty" => penalty,
        ),
        "provenance" => Dict("adapter" => "deterministic_fixture",
                             "source_stage" => stage),
        "transition_delta_report" => Dict("coordinate_space" => "effective_named",
                                          "candidate_class" => "archive_transfer",
                                          "max_abs_delta" => 0.02),
    )
end

function write_stage_summary(stage_root, report, gate, deltas)
    summary = Dict{String,Any}(
        "stage" => basename(stage_root),
        "archive_count" => report["archive_count"],
        "scalar_best" => report["scalar_best"]["candidate"],
        "quality_band" => report["quality_band"],
        "diversity" => report["diversity"],
        "transition_deltas" => deltas,
        "extension_gate" => gate,
    )
    O.safe_save_json(joinpath(stage_root, "stage_summary.json"), summary;
                     label="fixture_stage_summary")
    return summary
end

@testset "deterministic two-stage archive handoff" begin
    root = mktempdir()
    short_root = joinpath(root, "short")
    long_root = joinpath(root, "long")
    mkpath(short_root)

    entries = [
        candidate("survivor-1", 1.00, [0.10, 0.90]; weekly=0.90, cumulative=0.90, age=0.90),
        candidate("survivor-2", 1.01, [0.90, 0.10]; weekly=0.10, cumulative=1.20, age=1.20),
        candidate("survivor-3", 1.02, [0.20, 0.20]; weekly=1.20, cumulative=0.10, age=1.20),
        candidate("survivor-4", 1.03, [0.80, 0.80]; weekly=1.20, cumulative=1.20, age=0.10),
        candidate("failed", 0.1, [0.5, 0.5]; status="failed"),
        candidate("wrong-stage", 0.1, [0.5, 0.5]; stage="other"),
        candidate("wrong-horizon", 0.1, [0.5, 0.5]; months=4),
        candidate("nonfinite", Inf, [0.5, NaN]),
    ]
    report = O.survivor_archive_update(Any[], entries;
        current_stage="short", current_fit_months=3, target_size=3,
        min_distance=0.05, return_report=true)
    @test report["archive_count"] >= 3
    @test report["rejected_counts"]["status"] == 1
    @test report["rejected_counts"]["stage"] == 1
    @test report["rejected_counts"]["horizon"] == 1
    @test report["rejected_counts"]["nonfinite"] == 1
    @test all(x["status"] == "completed" && x["stage"] == "short" &&
              x["fit_months"] == 3 for x in report["archive"])

    archive_path = joinpath(short_root, "survivor_archive.json")
    O.safe_save_json(archive_path, report["archive"]; label="fixture_archive")
    manifest_path = O.persist_archive_transfer_manifest(short_root, report["archive"];
        archive_path=archive_path, stage="short", fit_months=3)
    loaded = O.load_transfer_survivor_archive(root, "long";
        expected_fit_months=3, expected_manifest_path=manifest_path,
        stage_order=["short", "long"])
    @test [x["candidate"] for x in loaded] ==
          [x["candidate"] for x in report["archive"]]
    @test loaded == JSON.parsefile(archive_path)

    gate = O.archive_quality_gate(loaded; current_stage="short",
        current_fit_months=3, current_best_score=1.0,
        current_quality_band=Dict("threshold" => 1.10),
        current_minimum_size=3, current_diversity_passed=true,
        target_size=3)
    @test gate["status"] == "passed"
    @test gate["admitted_count"] == length(loaded)

    transfer_slots = [x["candidate"] for x in loaded]
    immigrants = ["immigrant-1", "immigrant-2"]
    transfer_manifest = Dict{String,Any}(
        "source_archive_path" => manifest_path,
        "source_stage" => "short",
        "target_stage" => "long",
        "protected_transfer_slots" => transfer_slots,
        "immigrant_slots" => immigrants,
        "candidate_order" => vcat(transfer_slots, immigrants),
    )
    O.safe_save_json(joinpath(short_root, "transfer_manifest.json"), transfer_manifest;
                     label="fixture_transfer_manifest")
    @test transfer_manifest["protected_transfer_slots"] == transfer_slots
    @test isempty(intersect(Set(transfer_slots), Set(immigrants)))
    @test first(transfer_slots) != report["scalar_best"]["candidate"] ||
          length(transfer_slots) > 1

    deltas = [Dict("candidate" => id, "max_abs_delta" => 0.02,
                   "coordinate_space" => "effective_named")
              for id in transfer_slots]
    summary = write_stage_summary(short_root, report, gate, deltas)
    @test summary["archive_count"] == length(transfer_slots)
    @test haskey(summary, "quality_band") && haskey(summary, "diversity")
    @test length(summary["transition_deltas"]) == length(transfer_slots)

    # A passing handoff is allowed to create the next-stage root, but it must
    # consume exactly the canonical immediate-predecessor archive.
    mkpath(long_root)
    O.safe_save_json(joinpath(long_root, "transfer_candidates.json"),
                     loaded; label="fixture_transfer_candidates")
    @test JSON.parsefile(joinpath(long_root, "transfer_candidates.json")) == loaded
end

@testset "blocked quality gate leaves no next-stage artifacts" begin
    root = mktempdir()
    short_root = joinpath(root, "short")
    mkpath(short_root)
    archive = [candidate("poor", 2.0, [0.1, 0.1])]
    O.safe_save_json(joinpath(short_root, "survivor_archive.json"), archive;
                     label="fixture_blocked_archive")
    before = sort(readdir(root))
    gate = O.archive_quality_gate(archive; current_stage="short",
        current_fit_months=3, current_best_score=2.0,
        current_quality_band=Dict("threshold" => 1.0),
        current_minimum_size=1, current_diversity_passed=true, target_size=1)
    blocked = merge(gate, Dict{String,Any}(
        "status" => "blocked", "next_stage_created" => false,
        "reason" => gate["refusal_reason"], "stage" => "short"))
    blocked_path = joinpath(short_root, "stage_blocked.json")
    O.safe_save_json(blocked_path, blocked; label="fixture_stage_blocked")
    @test gate["status"] == "blocked"
    @test gate["refusal_reason"] == "current_objective_quality_failed"
    @test JSON.parsefile(blocked_path)["admitted_count"] == 1
    @test !isdir(joinpath(root, "long"))
    @test before == sort(readdir(root))
end

@testset "archive edge inputs fail closed and deduplicate" begin
    root = mktempdir()
    empty_report = O.survivor_archive_update(Any[], Any[];
        current_stage="short", current_fit_months=3, return_report=true)
    @test empty_report["archive_count"] == 0
    @test empty_report["adaptive_target_status"] == "constrained"

    duplicate = candidate("same", 1.0, [0.1, 0.1])
    duplicate_report = O.survivor_archive_update(Any[], [duplicate, deepcopy(duplicate)];
        current_stage="short", current_fit_months=3, return_report=true)
    @test duplicate_report["archive_count"] == 1
    @test duplicate_report["rejected_counts"]["duplicate"] == 1

    stage = joinpath(root, "short")
    mkpath(stage)
    archive_path = joinpath(stage, "survivor_archive.json")
    O.safe_save_json(archive_path, [duplicate]; label="fixture_malformed_archive")
    manifest_path = O.persist_archive_transfer_manifest(stage, [duplicate];
        archive_path=archive_path, stage="short", fit_months=3)
    open(manifest_path, "w") do io
        write(io, "{ malformed archive manifest")
    end
    evidence = O.load_transfer_survivor_archive(root, "long";
        expected_manifest_path=manifest_path, stage_order=["short", "long"],
        return_evidence=true)
    @test evidence["status"] == "rejected"
    @test evidence["failure_class"] in ("malformed_manifest", "archive_loader_exception")
    @test !isdir(joinpath(root, "long"))
end
