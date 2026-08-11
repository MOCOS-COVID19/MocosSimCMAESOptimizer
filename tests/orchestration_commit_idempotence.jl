using Test
using JSON
using SHA

include(joinpath(@__DIR__, "..", "src", "MocosSimCMAESOptimizer.jl"))
const O = MocosSimCMAESOptimizer

function write_committed_fixture(root)
    stage_root = joinpath(root, "stage_short")
    mkpath(stage_root)
    commit = joinpath(stage_root, "iter_1", "iteration_commit.json")
    if isfile(commit)
        return stage_root, false
    end
    mkpath(dirname(commit))
    names = ["beta[1]", "beta[2]"]
    rows = [Dict("stage"=>"stage_short", "iteration"=>1, "candidate"=>i,
                 "status"=>"completed", "score"=>0.1 * i,
                 "parameter_names"=>names) for i in 1:3]
    O.safe_save_json(joinpath(stage_root, "stage_state.json"),
        Dict("stage"=>"stage_short", "iteration"=>1, "best_candidate_id"=>1,
             "best_score"=>0.1, "param_names"=>names,
             "completed_count"=>3, "failed_count"=>0, "skipped_count"=>0,
             "pending_count"=>0, "cma_update_count"=>1))
    open(joinpath(stage_root, "iter_metrics.jsonl"), "w") do io
        for row in rows
            JSON.print(io, row); print(io, '\n')
        end
    end
    O.safe_save_json(joinpath(stage_root, "top_candidates.json"), rows)
    O.safe_save_json(joinpath(stage_root, "survivor_archive.json"), rows[1:2])
    O.safe_save_json(joinpath(stage_root, "full_reusable_state.json"),
        Dict("stage"=>"stage_short", "iteration"=>1, "archive_ids"=>[1, 2],
             "param_names"=>names, "mean"=>[0.1, 0.2],
             "covariance"=>[[1.0, 0.0], [0.0, 1.0]], "cma_update_count"=>1))
    committed = ["stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
        "survivor_archive.json", "full_reusable_state.json"]
    hashes = Dict(name => bytes2hex(SHA.sha256(read(joinpath(stage_root, name))))
                  for name in committed)
    O.safe_save_json(commit, Dict("status"=>"committed", "stage"=>"stage_short",
        "iteration"=>1, "candidate_ids"=>[1, 2, 3], "cma_update_count"=>1,
        "schema_version"=>"fixture-v1",
        "artifact_hashes"=>hashes, "artifact_key_set"=>committed,
        "artifact_hash_manifest"=>bytes2hex(SHA.sha256(JSON.json(hashes))),
        "artifact_integrity_digest"=>bytes2hex(SHA.sha256(JSON.json(hashes)))))
    return stage_root, true
end

@testset "committed artifact schema cannot downgrade" begin
    root = mktempdir()
    stage_root, _ = write_committed_fixture(root)
    commit_path = joinpath(stage_root, "iter_1", "iteration_commit.json")
    commit = O.load_json(commit_path)

    # A fixture contract is explicit, while omission is never inferred.
    @test O.validate_committed_artifacts(stage_root, "fixture-v1")["valid"]
    delete!(commit, "schema_version")
    O.safe_save_json(commit_path, commit)
    missing_schema = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !missing_schema["valid"]
    @test any(c -> occursin("schema", lowercase(c)), missing_schema["contradictions"])

    # A fixture discriminator must not be accepted for a production-shaped
    # manifest with a missing production artifact.
    commit["schema_version"] = "fixture-v1"
    commit["artifact_key_set"] = vcat(commit["artifact_key_set"], "survivor_archive_summary.json")
    commit["artifact_hashes"]["survivor_archive_summary.json"] = "missing"
    O.safe_save_json(commit_path, commit)
    downgraded = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !downgraded["valid"]
    @test any(c -> occursin("production", lowercase(c)) || occursin("artifact", lowercase(c)),
              downgraded["contradictions"])

    commit["schema_version"] = "unknown-v99"
    O.safe_save_json(commit_path, commit)
    unknown = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !unknown["valid"]
end

@testset "production loader rejects recomputed fixture downgrade" begin
    root = mktempdir()
    stage_root, _ = write_committed_fixture(root)
    commit_path = joinpath(stage_root, "iter_1", "iteration_commit.json")
    commit = O.load_json(commit_path)
    production_keys = [
        "stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
        "survivor_archive.json", "survivor_archive_summary.json",
        "full_reusable_state.json", "archive_transfer_manifest.json",
        "iter_1/candidate_list.txt", "iter_1/cma_sampling_state.json",
        "iter_1/top_candidates.json",
    ]
    for relative in production_keys
        path = joinpath(stage_root, relative)
        mkpath(dirname(path))
        isfile(path) || write(path, "{}")
    end
    hashes = Dict{String,Any}(
        relative => bytes2hex(SHA.sha256(read(joinpath(stage_root, relative))))
        for relative in production_keys
    )
    commit["schema_version"] = "production-v1"
    commit["artifact_key_set"] = production_keys
    commit["artifact_hashes"] = hashes
    commit["artifact_hash_manifest"] = bytes2hex(SHA.sha256(JSON.json(hashes)))
    commit["artifact_integrity_digest"] = commit["artifact_hash_manifest"]
    O.safe_save_json(commit_path, commit)
    @test O.validate_committed_artifacts(stage_root, "production-v1")["valid"]

    # Forge a fixture-shaped downgrade, including fresh hashes and digests.
    for relative in setdiff(production_keys, [
        "stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
        "survivor_archive.json", "full_reusable_state.json",
    ])
        rm(joinpath(stage_root, relative))
    end
    fixture_keys = [
        "stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
        "survivor_archive.json", "full_reusable_state.json",
    ]
    fixture_hashes = Dict{String,Any}(
        relative => bytes2hex(SHA.sha256(read(joinpath(stage_root, relative))))
        for relative in fixture_keys
    )
    commit["schema_version"] = "fixture-v1"
    commit["artifact_key_set"] = fixture_keys
    commit["artifact_hashes"] = fixture_hashes
    commit["artifact_hash_manifest"] = bytes2hex(SHA.sha256(JSON.json(fixture_hashes)))
    commit["artifact_integrity_digest"] = commit["artifact_hash_manifest"]
    O.safe_save_json(commit_path, commit)
    @test O.validate_committed_artifacts(stage_root, "fixture-v1")["valid"]
    forged = O.validate_committed_artifacts(stage_root, "production-v1")
    @test !forged["valid"]
    @test any(occursin("context", lowercase(c)) || occursin("schema", lowercase(c))
              for c in forged["contradictions"])
end

function snapshot(paths)
    Dict(path => bytes2hex(SHA.sha256(read(path))) for path in paths)
end

@testset "committed stage rerun is idempotent" begin
    root = mktempdir()
    stage_root, first_write = write_committed_fixture(root)
    @test first_write
    first = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test first["valid"]
    @test isempty(first["contradictions"])
    files = sort(filter(isfile, [joinpath(stage_root, n) for n in
        ("stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
         "survivor_archive.json", "full_reusable_state.json",
         "iter_1/iteration_commit.json")]))
    before = snapshot(files)
    stage_root2, second_write = write_committed_fixture(root)
    @test stage_root2 == stage_root
    @test !second_write
    after = snapshot(files)
    @test before == after
    second = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test second["candidate_ids"] == first["candidate_ids"]
    @test second["artifact_hashes"] == first["artifact_hashes"]
    @test second["jsonl_record_count"] == 3
end

@testset "cross-file join reports contradictions explicitly" begin
    root = mktempdir()
    stage_root, _ = write_committed_fixture(root)
    path = joinpath(stage_root, "top_candidates.json")
    rows = O.load_json(path)
    rows[1]["status"] = "failed"
    O.safe_save_json(path, rows)
    report = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !report["valid"]
    @test any(occursin("status", c) || occursin("contradiction", c)
              for c in report["contradictions"])
end

@testset "committed artifact manifest rejects extras and tampering" begin
    root = mktempdir()
    stage_root, _ = write_committed_fixture(root)
    commit_path = joinpath(stage_root, "iter_1", "iteration_commit.json")
    commit = O.load_json(commit_path)
    committed = [
        "stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
        "survivor_archive.json", "full_reusable_state.json",
    ]
    hashes = Dict{String,Any}(name => bytes2hex(SHA.sha256(read(joinpath(stage_root, name))))
                              for name in committed)
    commit["artifact_hashes"] = hashes
    commit["artifact_key_set"] = committed
    commit["artifact_hash_manifest"] = bytes2hex(SHA.sha256(JSON.json(hashes)))
    O.safe_save_json(commit_path, commit)
    @test O.validate_committed_artifacts(stage_root, "fixture-v1")["valid"]

    commit = O.load_json(commit_path)
    commit["artifact_hashes"]["unexpected.json"] = bytes2hex(SHA.sha256(UInt8[]))
    commit["artifact_key_set"] = vcat(commit["artifact_key_set"], ["unexpected.json"])
    write(joinpath(stage_root, "unexpected.json"), "{}")
    O.safe_save_json(commit_path, commit)
    extra = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !extra["valid"]
    @test any(occursin("unexpected", c) || occursin("artifact key", c)
              for c in extra["contradictions"])
    rm(joinpath(stage_root, "unexpected.json"))
    commit = O.load_json(commit_path)
    delete!(commit["artifact_hashes"], "top_candidates.json")
    commit["artifact_key_set"] = filter(!=("top_candidates.json"), commit["artifact_key_set"])
    commit["artifact_hash_manifest"] =
        bytes2hex(SHA.sha256(JSON.json(commit["artifact_hashes"])))
    commit["artifact_integrity_digest"] = commit["artifact_hash_manifest"]
    O.safe_save_json(commit_path, commit)
    missing = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !missing["valid"]
    @test any(occursin("missing or extra", c) || occursin("required key", c)
              for c in missing["contradictions"])
    # Restore the committed manifest before testing content tampering.
    commit = O.load_json(commit_path)
    commit["artifact_hashes"] = hashes
    commit["artifact_key_set"] = committed
    commit["artifact_hash_manifest"] = bytes2hex(SHA.sha256(JSON.json(commit["artifact_hashes"])))
    commit["artifact_integrity_digest"] = commit["artifact_hash_manifest"]
    O.safe_save_json(commit_path, commit)

    open(joinpath(stage_root, "survivor_archive.json"), "a") do io
        print(io, "\n")
    end
    tampered = O.validate_committed_artifacts(stage_root, "fixture-v1")
    @test !tampered["valid"]
    @test any(occursin("hash", lowercase(c)) for c in tampered["contradictions"])
end
