using Test
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function evaluation(stage, iteration, candidate, score, vector; explicit=true)
    value = Dict{String,Any}(
        "stage" => stage, "iteration" => iteration, "candidate" => candidate,
        "status" => "completed", "fit_months" => 3, "score" => score,
        "evaluated_vector" => vector, "parameter_names" => ["x", "y"])
    explicit && (value["archive_entry_id"] = O.archive_entry_id(stage, iteration, candidate))
    value
end

@testset "archive evaluation identity" begin
    first = evaluation("short", 1, 7, 1.0, [0.0, 0.0])
    later = evaluation("short", 2, 7, 1.01, [1.0, 1.0])
    report = O.survivor_archive_update(Any[], [first, later];
        target_size=2, max_size=2, min_distance=0.0, return_report=true)
    @test Set(O.archive_entry_id.(report["archive"])) == Set(["short:1:7", "short:2:7"])

    duplicate = O.survivor_archive_update(Any[], [first, deepcopy(first)];
        target_size=2, max_size=2, min_distance=0.0, return_report=true)
    @test duplicate["archive_count"] == 1
    @test duplicate["rejected_counts"]["duplicate"] == 1

    other_stage = evaluation("long", 1, 7, 1.02, [0.5, 0.5])
    stages = O.survivor_archive_update(Any[], [first, other_stage];
        target_size=2, max_size=2, min_distance=0.0)
    @test Set(O.archive_entry_id.(stages)) == Set(["short:1:7", "long:1:7"])
end

@testset "manifest identity round trip and legacy compatibility" begin
    root = mktempdir()
    stage_root = joinpath(root, "short")
    mkpath(stage_root)
    values = [evaluation("short", 1, 3, 1.0, [0.0, 1.0]),
              evaluation("short", 2, 3, 1.01, [1.0, 0.0])]
    O.safe_save_json(joinpath(stage_root, "survivor_archive.json"), values)
    manifest_path = O.persist_archive_transfer_manifest(stage_root, values;
        stage="short", fit_months=3)
    manifest = JSON.parsefile(manifest_path)
    @test manifest["admitted_ids"] == ["short:1:3", "short:2:3"]
    @test O.load_transfer_survivor_archive(root, "long";
        expected_fit_months=3, expected_manifest_path=manifest_path,
        stage_order=["short", "long"]) == values

    for legacy_candidate in (9, "candidate-nine")
        legacy = [evaluation("short", 1, legacy_candidate, 1.0, [0.2, 0.8]; explicit=false)]
        O.safe_save_json(joinpath(stage_root, "survivor_archive.json"), legacy)
        legacy_manifest = O.persist_archive_transfer_manifest(stage_root, legacy;
            stage="short", fit_months=3)
        @test JSON.parsefile(legacy_manifest)["admitted_ids"] == [string(legacy_candidate)]
        @test !isempty(O.load_transfer_survivor_archive(root, "long";
            expected_fit_months=3, expected_manifest_path=legacy_manifest,
            stage_order=["short", "long"]))
        if legacy_candidate isa Integer
            numeric_manifest = JSON.parsefile(legacy_manifest)
            numeric_manifest["admitted_ids"] = [legacy_candidate]
            numeric_manifest["admitted_order"] = [legacy_candidate]
            O.safe_save_json(legacy_manifest, numeric_manifest)
            @test !isempty(O.load_transfer_survivor_archive(root, "long";
                expected_fit_months=3, expected_manifest_path=legacy_manifest,
                stage_order=["short", "long"]))
        end
    end

    O.safe_save_json(joinpath(stage_root, "survivor_archive.json"), values)
    O.persist_archive_transfer_manifest(stage_root, values; stage="short", fit_months=3)
    tampered = JSON.parsefile(manifest_path)
    reverse!(tampered["admitted_order"])
    O.safe_save_json(manifest_path, tampered)
    @test isempty(O.load_transfer_survivor_archive(root, "long";
        expected_fit_months=3, expected_manifest_path=manifest_path,
        stage_order=["short", "long"]))
end
