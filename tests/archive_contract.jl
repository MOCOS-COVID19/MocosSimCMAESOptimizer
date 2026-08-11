using Test
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function fixture(id, score, vector; status="completed", stage="short", months=3)
    Dict{String,Any}(
        "candidate" => id, "score" => score, "evaluated_vector" => vector,
        "parameter_names" => ["p[1]", "p[2]"], "status" => status,
        "stage" => stage, "fit_months" => months,
        "metrics" => Dict("weekly_control_score" => score,
            "daily_detections_cumulative" => score,
            "daily_age_05_14_detections" => score,
            "temporal_jump_penalty" => score),
        "provenance" => Dict("source" => "fixture"),
    )
end

@testset "adaptive archive admits only current stage and reports policy" begin
    entries = [fixture("a", 1.0, [0.1, 0.1]),
               fixture("b", 1.01, [0.9, 0.9]),
               fixture("failed", 0.1, [0.2, 0.2]; status="failed"),
               fixture("old", 0.1, [0.3, 0.3]; stage="old")]
    result = O.survivor_archive_update(Any[], entries; target_size=40,
        current_stage="short", current_fit_months=3, return_report=true)
    @test result["archive_count"] == 2
    @test result["configured_target"] == 40
    @test result["technical_cap"] == 200
    @test all(get(x, "stage", "") == "short" for x in result["archive"])
    @test result["rejected_counts"]["status"] == 1
    @test result["rejected_counts"]["stage"] == 1
    @test isfinite(result["quality_band"]["threshold"])
end

@testset "normalized distance keeps boundary and drops duplicates" begin
    a = fixture("a", 1.0, [0.0, 0.0])
    b = fixture("b", 1.01, [0.1414213562, 0.0])
    c = fixture("c", 1.02, [0.151, 0.0])
    b["metrics"]["weekly_control_score"] = 0.5
    c["metrics"]["weekly_control_score"] = 0.4
    result = O.survivor_archive_update(Any[], [a, b, c];
        target_size=40, min_distance=0.1, return_report=true)
    @test "a" in [x["candidate"] for x in result["archive"]]
    @test length(result["archive"]) >= 2
end

@testset "quality gate blocks insufficient current objective" begin
    archive = [fixture("a", 1.0, [0.1, 0.1])]
    gate = O.archive_quality_gate(archive; current_stage="short",
        current_fit_months=3, current_best_score=2.0, target_size=40,
        minimum_size=2)
    @test gate["status"] == "blocked"
    @test gate["next_stage_created"] == false
    @test haskey(gate, "refusal_reason")
end
