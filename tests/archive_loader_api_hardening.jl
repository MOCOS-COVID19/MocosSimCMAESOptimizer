using Test
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function loader_entry(id)
    Dict{String,Any}(
        "candidate" => id, "score" => 1.0, "status" => "completed",
        "stage" => "short", "fit_months" => 3,
        "evaluated_vector" => [1.0], "parameter_names" => ["p[1]"],
    )
end

@testset "archive loader requires authoritative predecessor derivation" begin
    root = mktempdir()
    stage = joinpath(root, "short")
    mkpath(stage)
    values = [loader_entry("a")]
    archive_path = joinpath(stage, "survivor_archive.json")
    O.safe_save_json(archive_path, values)
    manifest_path = O.persist_archive_transfer_manifest(stage, values;
        stage="short", fit_months=3)

    @test isempty(O.load_transfer_survivor_archive(root, "long";
        predecessor_stage="short", expected_fit_months=3))
    evidence = O.load_transfer_survivor_archive(root, "long";
        predecessor_stage="short", expected_fit_months=3, return_evidence=true)
    @test evidence["status"] == "rejected"
    @test evidence["failure_class"] == "missing_authoritative_stage_order"

    @test O.load_transfer_survivor_archive(root, "long";
        stage_order=["short", "long"], expected_fit_months=3,
        expected_manifest_path=manifest_path) == values
    @test O.load_transfer_survivor_archive(root, "long";
        configuration=Dict("stages" => [Dict("name" => "short"),
                                        Dict("name" => "long")]),
        expected_fit_months=3, expected_manifest_path=manifest_path) == values
end

@testset "archive loader rejects schema and adjacency metadata with evidence" begin
    root = mktempdir()
    stage = joinpath(root, "short")
    mkpath(stage)
    values = [loader_entry("a")]
    archive_path = joinpath(stage, "survivor_archive.json")
    O.safe_save_json(archive_path, values)
    manifest_path = O.persist_archive_transfer_manifest(stage, values;
        stage="short", fit_months=3)
    manifest = JSON.parsefile(manifest_path)

    for (field, value, reason) in [
        ("schema_version", "archive-transfer-v1", "schema_version_mismatch"),
        ("source_stage", "unrelated", "source_stage_not_immediate_predecessor"),
        ("horizon", 4, "horizon_mismatch"),
    ]
        corrupted = copy(manifest)
        corrupted[field] = value
        O.safe_save_json(manifest_path, corrupted)
        evidence = O.load_transfer_survivor_archive(root, "long";
            stage_order=["short", "long"], expected_fit_months=3,
            return_evidence=true)
        @test evidence["status"] == "rejected"
        @test evidence["failure_class"] == reason
    end
end
