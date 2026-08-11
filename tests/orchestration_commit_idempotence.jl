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
    O.safe_save_json(commit, Dict("status"=>"committed", "stage"=>"stage_short",
        "iteration"=>1, "candidate_ids"=>[1, 2, 3], "cma_update_count"=>1))
    return stage_root, true
end

function snapshot(paths)
    Dict(path => bytes2hex(SHA.sha256(read(path))) for path in paths)
end

@testset "committed stage rerun is idempotent" begin
    root = mktempdir()
    stage_root, first_write = write_committed_fixture(root)
    @test first_write
    first = O.validate_committed_artifacts(stage_root)
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
    second = O.validate_committed_artifacts(stage_root)
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
    report = O.validate_committed_artifacts(stage_root)
    @test !report["valid"]
    @test any(occursin("status", c) || occursin("contradiction", c)
              for c in report["contradictions"])
end
