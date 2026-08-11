using Test
using JSON

include(joinpath(@__DIR__, "..", "src", "MocosSimCMAESOptimizer.jl"))
const O = MocosSimCMAESOptimizer

function marker_fixture(n, successful)
    root = mktempdir()
    list = joinpath(root, "candidates.txt")
    dirs = String[]
    for i in 1:n
        d = joinpath(root, "cand_$(lpad(i, 2, '0'))")
        mkpath(d)
        push!(dirs, d)
    end
    for i in successful
        touch(joinpath(dirs[i], "done.ok"))
    end
    open(list, "w") do io
        foreach(d -> println(io, d), dirs)
    end
    return root, list, dirs
end

@testset "orchestration terminal accounting" begin
    for (count, expected) in ((8, false), (9, true), (10, true))
        root, list, dirs = marker_fixture(10, 1:count)
        result = O.wait_for_iteration_outputs(list; poll=0.001,
            min_completion_fraction=0.9, finish_iter_delay=0, max_wait=0.05)
        @test result["threshold_reached"] == expected
        @test result["done"] == count
        @test result["failed"] == 0
        @test result["skipped"] == 10 - count
        @test result["pending_count"] == 0
        @test length(result["statuses"]) == 10
        @test all(get(s, "status", "") in ("completed", "failed", "skipped")
                  for s in result["statuses"])
    end
end

@testset "marker conflicts fail closed" begin
    root, list, dirs = marker_fixture(1, [1])
    touch(joinpath(dirs[1], "failed.ok"))
    result = O.wait_for_iteration_outputs(list; poll=0.001,
        min_completion_fraction=0.9, finish_iter_delay=0)
    @test result["done"] == 0
    @test result["failed"] == 1
    @test result["statuses"][1]["failure_class"] == "marker_conflict"
end

@testset "resume requires committed artifacts" begin
    root = mktempdir()
    stage = joinpath(root, "stage")
    iter = joinpath(stage, "iter_1")
    mkpath(joinpath(iter, "cand_01"))
    open(joinpath(iter, "candidate_list.txt"), "w") do io
        println(io, joinpath(iter, "cand_01"))
    end
    info = O.stage_resume_info(stage)
    @test info["iteration_completed"] == false
    @test info["last_iter"] == 1
    @test info["resume_iteration"] == 1
end

@testset "resume selects newest incomplete iteration" begin
    root = mktempdir()
    stage = joinpath(root, "stage")
    mkpath(joinpath(stage, "iter_1"))
    for name in ("top_candidates.json", "stage_state.json",
                 "full_reusable_state.json", "iter_metrics.jsonl")
        open(joinpath(stage, name), "w") do io
            print(io, name == "iter_metrics.jsonl" ? "" : "{}")
        end
    end
    open(joinpath(stage, "iter_1", "candidate_list.txt"), "w") do io end
    mkpath(joinpath(stage, "iter_2", "cand_01"))
    open(joinpath(stage, "iter_2", "candidate_list.txt"), "w") do io end
    info = O.stage_resume_info(stage)
    @test info["last_iter"] == 2
    @test info["iteration_completed"] == false
    @test info["resume_iteration"] == 2
end

@testset "resume requires matching atomic iteration commit" begin
    root = mktempdir()
    stage = joinpath(root, "stage_a")
    iter = joinpath(stage, "iter_1")
    mkpath(iter)
    open(joinpath(iter, "candidate_list.txt"), "w") do io end
    for name in ("top_candidates.json", "stage_state.json",
                 "full_reusable_state.json", "iter_metrics.jsonl")
        open(joinpath(name == "stage_state.json" || name == "full_reusable_state.json" ||
                      name == "iter_metrics.jsonl" ? stage : iter, name), "w") do io
            print(io, name == "iter_metrics.jsonl" ? "" : "{}")
        end
    end
    O.safe_save_json(joinpath(iter, "iteration_commit.json"),
        Dict("status" => "committed", "stage" => "wrong_stage", "iteration" => 1))
    @test O.stage_resume_info(stage)["iteration_completed"] == false
    O.safe_save_json(joinpath(iter, "iteration_commit.json"),
        Dict("status" => "committed", "stage" => "stage_a", "iteration" => 1))
    @test O.stage_resume_info(stage)["iteration_completed"] == true
end

@testset "transition rejection has terminal artifacts" begin
    root = mktempdir()
    status = O.materialize_terminal_candidate!(root, "failed";
        failure_class="transition_policy_reject",
        details=Dict("stage" => "short", "iteration" => 2))
    @test status["status"] == "failed"
    @test isfile(joinpath(root, "failed.ok"))
    @test isfile(joinpath(root, "status.json"))
    @test O.load_json(joinpath(root, "status.json"))["failure_class"] ==
        "transition_policy_reject"
end

@testset "fresh candidate roots never collide" begin
    root = mktempdir()
    first = O.create_candidate_root(root, "candidate")
    second = O.create_candidate_root(root, "candidate")
    @test first != second
    @test isdir(first) && isdir(second)
end
