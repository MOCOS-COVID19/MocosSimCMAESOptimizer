using Test
using JSON

include(joinpath(@__DIR__, "..", "src", "MocosSimCMAESOptimizer.jl"))
const O = MocosSimCMAESOptimizer

function fresh_candidates(root, n)
    dirs = [joinpath(root, "candidate_$(lpad(i, 2, '0'))") for i in 1:n]
    mkpath.(dirs)
    list = joinpath(root, "candidate_list.txt")
    open(list, "w") do io
        foreach(d -> println(io, d), dirs)
    end
    return dirs, list
end

@testset "late terminal markers are observed during grace" begin
    mktempdir() do root
        dirs, list = fresh_candidates(root, 10)
        # Eight completed candidates are below the 0.9 threshold.  Candidate
        # nine arrives after polling starts, while candidate ten remains
        # pending and is materialized as skipped after grace.
        for d in dirs[1:8]
            touch(joinpath(d, "done.ok"))
        end
        @async begin
            sleep(0.1)
            touch(joinpath(dirs[9], "done.ok"))
        end

        result = O.wait_for_iteration_outputs(list; poll=0.005,
            min_completion_fraction=0.9, finish_iter_delay=1, max_wait=2.0)
        @test result["threshold_reached"] == true
        @test result["done"] == 9
        @test result["failed"] == 0
        @test result["skipped"] == 1
        @test result["pending_count"] == 0
        @test result["iteration_truncated"] == true
        @test O.load_json(joinpath(dirs[9], "status.json"))["status"] == "completed"
        @test O.load_json(joinpath(dirs[10], "status.json"))["status"] == "skipped"

        O.safe_save_json(joinpath(root, "late_marker_evidence.json"),
            Dict("threshold_done" => 9, "observed_done" => result["done"],
                 "observed_skipped" => result["skipped"],
                 "iteration_truncated" => result["iteration_truncated"]))
    end
end

function archive_entry(id, status, score, vector; stage="short", months=3)
    return Dict{String,Any}(
        "candidate" => id, "id" => id, "status" => status, "stage" => stage,
        "fit_months" => months, "score" => score,
        "evaluated_vector" => vector,
        "metrics" => Dict("daily" => score),
        "provenance" => Dict("source" => "fixture", "candidate_id" => id),
    )
end

@testset "mixed terminal statuses cannot enter ranking archive" begin
    mktempdir() do root
        dirs, _ = fresh_candidates(root, 5)
        for d in dirs
            touch(joinpath(d, "done.ok"))
        end
        O.materialize_terminal_candidate!(dirs[2], "failed";
            failure_class="missing_output")
        O.materialize_terminal_candidate!(dirs[3], "skipped";
            failure_class="iteration_truncated")
        O.materialize_terminal_candidate!(dirs[4], "failed";
            failure_class="malformed_output")

        terminal = O.normalized_iteration_result(dirs; min_completion_fraction=0.9)
        @test terminal["done"] == 2
        @test terminal["failed"] == 2
        @test terminal["skipped"] == 1
        @test terminal["pending_count"] == 0

        entries = [
            archive_entry("completed-good", "completed", 0.10, [0.1, 0.2]),
            archive_entry("completed-other", "completed", 0.20, [0.8, 0.9]),
            archive_entry("failed", "failed", 0.01, [0.0, 0.0]),
            archive_entry("skipped", "skipped", 0.02, [0.3, 0.3]),
            archive_entry("missing-output", "failed", Inf, [0.4, 0.4]),
        ]
        report = O.survivor_archive_update(Any[], entries;
            current_stage="short", current_fit_months=3,
            target_size=4, max_size=10, return_report=true)
        ids = [String(get(x, "candidate", "")) for x in report["archive"]]
        @test ids == ["completed-good", "completed-other"]
        @test !any(x -> x in ids, ["failed", "skipped", "missing-output"])
        @test report["rejected_counts"]["status"] == 3
        @test all(get(x, "status", "") == "completed" for x in report["archive"])

        O.safe_save_json(joinpath(root, "mixed_status_evidence.json"),
            Dict("terminal_counts" => Dict("completed" => terminal["done"],
                "failed" => terminal["failed"], "skipped" => terminal["skipped"]),
                "ranked_ids" => ids, "rejected_counts" => report["rejected_counts"]))
    end
end

function normalized_collection_fixture(root)
    dirs, _ = fresh_candidates(root, 4)
    O.materialize_terminal_candidate!(dirs[1], "completed";
        details=Dict("score" => 1.25, "simulated" => true))
    O.materialize_terminal_candidate!(dirs[2], "failed";
        failure_class="missing_output")
    O.materialize_terminal_candidate!(dirs[3], "skipped";
        failure_class="iteration_truncated")
    O.materialize_terminal_candidate!(dirs[4], "failed";
        failure_class="marker_conflict")
    return dirs
end

@testset "local and collected terminal payloads are normalized identically" begin
    mktempdir() do root
        local_root = joinpath(root, "local")
        collected_root = joinpath(root, "collected")
        local_dirs = normalized_collection_fixture(local_root)
        collected_dirs = normalized_collection_fixture(collected_root)
        local_payload = O.normalized_iteration_result(local_dirs;
            min_completion_fraction=0.9)
        collected_payload = O.normalized_iteration_result(collected_dirs;
            min_completion_fraction=0.9)

        normalize(payload) = begin
            result = deepcopy(payload)
            for status in result["statuses"]
                pop!(status, "candidate_dir", nothing)
            end
            sort!(result["statuses"]; by=s -> (s["status"], string(get(s, "failure_class", ""))))
            result
        end
        @test normalize(local_payload) == normalize(collected_payload)
        @test local_payload["done"] == 1
        @test local_payload["failed"] == 2
        @test local_payload["skipped"] == 1
        @test local_payload["pending_count"] == 0
        @test local_payload["threshold_reached"] == false
        @test local_payload["iteration_truncated"] == true

        O.safe_save_json(joinpath(root, "local_collected_normalized.json"),
            Dict("local" => normalize(local_payload),
                 "collected" => normalize(collected_payload),
                 "parity" => normalize(local_payload) == normalize(collected_payload),
                 "simulator_invoked" => false, "slurm_invoked" => false))
    end
end
