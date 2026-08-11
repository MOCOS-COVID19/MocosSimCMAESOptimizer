using Test
using Random
using LinearAlgebra
using SHA

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

const FIXTURE_PLAN = [
    ("short", 3), ("medium", 6), ("long", 12), ("target", 24)
]

function fixture_state(state::O.CMAState, rng)
    return Dict{String,Any}(
        "mean" => copy(state.mean), "sigma" => copy(state.sigma),
        "covariance" => copy(state.covariance), "p_c" => copy(state.p_c),
        "p_sigma" => copy(state.p_sigma), "rng_state" => O.rng_snapshot(rng),
    )
end

function run_fixture(root; resume::Bool=false, interrupt::Bool=true)
    mkpath(root)
    specs = [O.ParamSpec("curve", :temporal, 3, 0.0, 1.0),
             O.ParamSpec("bias", :scalar, 1, 0.0, 1.0)]
    previous_specs = [O.ParamSpec("curve", :temporal, 2, 0.0, 1.0),
                      O.ParamSpec("bias", :scalar, 1, 0.0, 1.0)]
    prior = O.CMAState([0.2, 0.4, 0.6], [0.05, 0.06, 0.07],
                       Matrix{Float64}(I, 3, 3), [0.1, 0.2, 0.3],
                       [0.3, 0.2, 0.1])
    next_state = O.stage_transition_state(
        prior, O.StageConfig("medium", 6, 2, 4, 0.1), specs;
        previous_specs=previous_specs)
    rng = resume ? O.restore_rng(O.load_json(joinpath(root, "resume_state.json"))["rng_state"]) :
                   MersenneTwister(913)
    start = resume ? O.load_json(joinpath(root, "resume_state.json"))["next_iteration"] : 1
    archive = resume ? O.load_json(joinpath(root, "resume_state.json"))["archive"] : Any[]
    population = Any[]
    for iteration in start:4
        population = [round.(rand(rng, length(next_state.mean)); digits=8) for _ in 1:4]
        for (i, vector) in enumerate(population)
            push!(archive, Dict{String,Any}("id" => "archive-$(iteration)-$(i)",
                "iteration" => iteration, "vector" => vector,
                "score" => sum(abs2, vector)))
        end
        if interrupt && !resume && iteration == 2
            O.safe_save_json(joinpath(root, "resume_state.json"),
                Dict("next_iteration" => 3, "archive" => archive,
                     "normalized_state" => fixture_state(next_state, rng),
                     "rng_state" => O.rng_snapshot(rng)))
            return Dict("interrupted" => true)
        end
    end
    sort!(archive; by=x -> (x["score"], x["id"]))
    result = Dict{String,Any}("population" => population, "archive" => archive,
        "normalized_state" => fixture_state(next_state, rng),
        "next_iteration" => 5)
    O.safe_save_json(joinpath(root, "final_state.json"), result)
    return result
end

@testset "transition orchestration fixtures" begin
    root = mktempdir()

    stages = Any[]
    previous_days = 0
    for (name, months) in FIXTURE_PLAN
        requested_days = months * 30
        effective_months = ceil(Int, requested_days / 30)
        effective_days = effective_months * 30
        push!(stages, Dict("stage" => name, "requested_months" => months,
            "requested_days" => requested_days, "effective_months" => effective_months,
            "effective_days" => effective_days, "remaining_days" => 720 - effective_days,
            "prefix_unchanged" => effective_days >= previous_days))
        previous_days = effective_days
    end
    plan = Dict("monthly_days" => 30, "target_months" => 24, "target_days" => 720,
        "policy" => "effective_months=ceil(requested_days/monthly_days)",
        "status" => "complete", "stages" => stages)
    plan_path = joinpath(root, "stage_plan_manifest.json")
    O.safe_save_json(plan_path, plan)
    @test plan["status"] == "complete"
    @test [x["effective_days"] for x in stages] == [90, 180, 360, 720]
    @test all(stages[i]["effective_days"] < stages[i+1]["effective_days"] for i in 1:3)
    @test all(x["prefix_unchanged"] for x in stages)
    @test isfile(plan_path)

    interrupted_root = joinpath(root, "interrupted")
    @test run_fixture(interrupted_root)["interrupted"] == true
    resume = run_fixture(interrupted_root; resume=true)
    uninterrupted = run_fixture(joinpath(root, "uninterrupted"); interrupt=false)
    @test resume["population"] == uninterrupted["population"]
    @test resume["archive"] == uninterrupted["archive"]
    @test resume["normalized_state"] == uninterrupted["normalized_state"]
    resume_path = joinpath(interrupted_root, "resume_state.json")
    @test O.load_json(resume_path)["rng_state"]["algorithm"] == "MersenneTwister"
    @test isfile(joinpath(interrupted_root, "final_state.json"))

    names = ["curve[1]", "curve[2]", "curve[3]", "bias[1]"]
    lock_hash = bytes2hex(SHA.sha256(codeunits(join(names[1:2], "|"))))
    lineage = Any[]
    for i in 1:4
        klass = i <= 2 ? "archive_transfer" : (i == 3 ? "immigrant/escape" : "new_dimension")
        push!(lineage, Dict("candidate_id" => "candidate-$i",
            "source_archive_id" => i <= 2 ? "archive-$i" : nothing,
            "candidate_class" => klass, "protected_transfer_slot" => i <= 2,
            "locked_coordinates" => names[1:2], "locked_prefix_hash" => lock_hash,
            "rng_algorithm" => "MersenneTwister", "rng_stream" => "transition-913",
            "transition_delta_report" => Dict("coordinate_space" => "effective",
                "version" => "v1", "max_abs_delta" => i <= 2 ? 0.05 : 0.0)))
    end
    lineage_path = joinpath(root, "transition_lineage.json")
    O.safe_save_json(lineage_path, lineage)
    @test count(x -> x["protected_transfer_slot"], lineage) == 2
    @test all(x -> x["source_archive_id"] !== nothing, lineage[1:2])
    @test all(x -> x["candidate_class"] != "archive_transfer", lineage[3:4])
    @test all(x -> x["locked_prefix_hash"] == lock_hash, lineage)
    @test isfile(lineage_path)

    equivalence = Dict("status" => "equivalent",
        "uninterrupted_archive_order" => [x["id"] for x in uninterrupted["archive"]],
        "resumed_archive_order" => [x["id"] for x in resume["archive"]],
        "population_equal" => resume["population"] == uninterrupted["population"],
        "archive_equal" => resume["archive"] == uninterrupted["archive"],
        "rng_state_equal" => resume["normalized_state"]["rng_state"] ==
                             uninterrupted["normalized_state"]["rng_state"])
    equivalence_path = joinpath(root, "uninterrupted_vs_resumed.json")
    O.safe_save_json(equivalence_path, equivalence)
    @test all(equivalence[k] for k in ("population_equal", "archive_equal", "rng_state_equal"))
    @test isfile(equivalence_path)
end
