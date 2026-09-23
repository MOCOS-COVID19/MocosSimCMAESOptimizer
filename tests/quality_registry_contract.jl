using Test
using JSON

include(joinpath(@__DIR__, "..", "src", "MocosSimCMAESOptimizer.jl"))
const O = MocosSimCMAESOptimizer

function qualified_entry(iteration, candidate, score, value, vector)
    metrics = Dict{String,Any}(
        "daily_detections" => value,
        "daily_detections_cumulative" => value,
        "daily_deaths" => value,
        "daily_deaths_cumulative" => value)
    O.evaluate_quality_gates!(metrics)
    return Dict{String,Any}(
        "stage" => "short", "iteration" => iteration,
        "candidate" => candidate, "status" => "completed",
        "fit_months" => 2, "score" => score,
        "evaluated_vector" => vector, "config" => Dict("p" => vector[1]),
        "provenance" => Dict("source" => "cma_population"),
        "metrics" => Dict("metrics" => metrics))
end

@testset "independent quality gates retain component evidence" begin
    metrics = Dict{String,Any}(
        "daily_detections" => 0.19,
        "daily_detections_cumulative" => 0.09,
        "daily_deaths" => 0.29,
        "daily_deaths_cumulative" => 0.11)
    result = O.evaluate_quality_gates!(metrics)
    @test result["achieved_levels"] == [0.5, 0.3]
    gate = only(filter(g -> g["threshold"] == 0.2, result["gates"]))
    @test gate["passed"] == false
    @test gate["components"]["detections_trajectory"]["passed"] == true
    @test gate["components"]["deaths_trajectory"]["value"] == 0.29
    @test metrics["quality_gates"] === result
end

@testset "qualified registry survives replacement, archive cap, and resume" begin
    mktempdir() do root
        path = joinpath(root, "qualified_candidate_registry.jsonl")
        first = qualified_entry(1, 1, 0.25, 0.25, [0.1, 0.1])
        better = qualified_entry(2, 1, 0.05, 0.05, [0.9, 0.9])
        @test length(O.append_qualified_candidate_registry!(path, [first];
            seeds=[42, 43], data_hash="data", calendar_hash="calendar")) == 1
        @test length(O.append_qualified_candidate_registry!(path, [better];
            seeds=[42, 43], data_hash="data", calendar_hash="calendar")) == 1

        registry = O.load_qualified_candidate_registry(path)
        @test length(registry) == 2
        @test length(unique(row["archive_entry_id"] for row in registry)) == 2
        @test all(row["used_seeds"] == [42, 43] for row in registry)
        @test all(row["horizon"] == 2 for row in registry)
        @test all(row["data_hash"] == "data" && row["calendar_hash"] == "calendar" for row in registry)

        limited = O.survivor_archive_update(Any[], registry;
            current_stage="short", current_fit_months=2,
            target_size=1, max_size=1, min_distance=0.0)
        @test length(limited) == 1
        @test length(O.load_qualified_candidate_registry(path)) == 2

        # A resumed process presents already-seen candidates again.  The file
        # remains append-only and unique rather than losing the older success.
        @test isempty(O.append_qualified_candidate_registry!(path, [first, better];
            seeds=[42, 43], data_hash="data", calendar_hash="calendar"))
        resumed = O.load_qualified_candidate_registry(path)
        @test length(resumed) == 2
        @test any(row["iteration"] == 1 for row in resumed)
    end
end
