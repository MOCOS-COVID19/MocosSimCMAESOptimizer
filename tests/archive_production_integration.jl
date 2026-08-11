using Test
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function entry(id, score; status="completed", stage="short", months=3)
    Dict{String,Any}(
        "candidate" => id, "score" => score, "status" => status,
        "stage" => stage, "fit_months" => months,
        "evaluated_vector" => [score, 1.0 - score],
        "parameter_names" => ["p[1]", "p[2]"],
    )
end

@testset "production archive admission uses independent quality inputs" begin
    archive = [entry("good", 1.0), entry("out", 1.4)]
    gate = O.archive_quality_gate(archive; current_stage="short",
        current_fit_months=3, current_best_score=1.0,
        current_quality_band=Dict("threshold" => 1.1),
        current_minimum_size=2, current_diversity_passed=true,
        target_size=2, minimum_size=2)
    @test gate["status"] == "blocked"
    @test gate["refusal_reason"] == "quality_band_failed"
end

@testset "archive transfer manifest is canonical and immediate" begin
    root = mktempdir()
    stage = joinpath(root, "short")
    mkpath(stage)
    values = [entry("a", 1.0), entry("b", 1.01)]
    archive_path = joinpath(stage, "survivor_archive.json")
    O.safe_save_json(archive_path, values)
    manifest_path = O.persist_archive_transfer_manifest(stage, values; fit_months=3)
    loaded = O.load_transfer_survivor_archive(root, "long";
        predecessor_stage="short", expected_fit_months=3,
        expected_manifest_path=manifest_path,
        stage_order=["short", "long"])
    @test loaded == values
    @test JSON.parsefile(manifest_path)["admitted_ids"] == ["a", "b"]
end

@testset "invalid archive status is never admissible" begin
    report = O.survivor_archive_update(Any[],
        [entry("bad", 0.1; status="failed")];
        current_stage="short", current_fit_months=3, return_report=true)
    @test report["archive_count"] == 0
    @test report["rejected_counts"]["status"] == 1
end

@testset "quality gate requires independent current-stage inputs" begin
    archive = [entry("good", 1.0), entry("also-good", 1.01)]
    missing_band = O.archive_quality_gate(archive; current_stage="short",
        current_fit_months=3, current_best_score=1.0,
        current_minimum_size=2, current_diversity_passed=true, target_size=2)
    @test missing_band["status"] == "blocked"
    @test missing_band["refusal_reason"] == "missing_external_quality_band"

    poor_current = O.archive_quality_gate(archive; current_stage="short",
        current_fit_months=3, current_best_score=2.0,
        current_quality_band=Dict("threshold" => 1.1),
        current_minimum_size=2, current_diversity_passed=true, target_size=2)
    @test poor_current["status"] == "blocked"
    @test poor_current["refusal_reason"] == "current_objective_quality_failed"
end

@testset "archive manifest identity and all predecessor fields are validated" begin
    root = mktempdir()
    stage = joinpath(root, "short")
    mkpath(stage)
    values = [entry("a", 1.0), entry("b", 1.01)]
    archive_path = joinpath(stage, "survivor_archive.json")
    O.safe_save_json(archive_path, values)
    manifest_path = O.persist_archive_transfer_manifest(stage, values;
        fit_months=3, stage="short")
    manifest = JSON.parsefile(manifest_path)
    @test haskey(manifest, "archive_id")
    @test haskey(manifest, "archive_hash")
    @test manifest["schema_version"] == "archive-transfer-v2"
    @test manifest["horizon"] == 3
    @test manifest["archive_count"] == 2
    @test O.load_transfer_survivor_archive(root, "long";
        predecessor_stage="short", expected_fit_months=3,
        expected_manifest_path=manifest_path,
        stage_order=["short", "long"]) == values

    manifest["admitted_order"] = ["b", "a"]
    O.safe_save_json(manifest_path, manifest)
    @test isempty(O.load_transfer_survivor_archive(root, "long";
        predecessor_stage="short", expected_fit_months=3,
        expected_manifest_path=manifest_path))
end

@testset "archive transfer rejects non-predecessor and malformed manifests without throwing" begin
    root = mktempdir()
    stage = joinpath(root, "short")
    mkpath(stage)
    values = [entry("a", 1.0), entry("b", 1.01)]
    archive_path = joinpath(stage, "survivor_archive.json")
    O.safe_save_json(archive_path, values)
    manifest_path = O.persist_archive_transfer_manifest(stage, values;
        fit_months=3, stage="short")

    # The authoritative order makes short the only predecessor of medium.
    @test O.load_transfer_survivor_archive(root, "medium";
        stage_order=["short", "medium", "long"], expected_fit_months=3) == values
    @test isempty(O.load_transfer_survivor_archive(root, "long";
        stage_order=["short", "medium", "long"], expected_fit_months=3))

    manifest = JSON.parsefile(manifest_path)
    for (field, bad) in [
        ("archive_count", "not-a-number"),
        ("horizon", nothing),
        ("admitted_ids", ["a"]),
        ("admitted_order", ["a", "b", "extra"]),
    ]
        corrupted = copy(manifest)
        corrupted[field] = bad
        O.safe_save_json(manifest_path, corrupted)
        @test isempty(O.load_transfer_survivor_archive(root, "medium";
            stage_order=["short", "medium", "long"], expected_fit_months=3))
    end
end
