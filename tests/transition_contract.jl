using Test
using LinearAlgebra
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

@testset "transition contract" begin
    specs_old = [O.ParamSpec("a", :scalar, 1, 0.0, 1.0),
                 O.ParamSpec("curve", :temporal, 2, 0.0, 1.0)]
    specs_new = [O.ParamSpec("curve", :temporal, 3, 0.0, 1.0),
                 O.ParamSpec("b", :scalar, 1, 0.0, 1.0),
                 O.ParamSpec("a", :scalar, 1, 0.0, 1.0)]
    prev = O.CMAState([0.2, 0.3, 0.4], [0.04, 0.05, 0.06],
                      Matrix{Float64}(I, 3, 3), [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    state = O.stage_transition_state(prev, O.StageConfig("next", 3, 1, 2, 0.1),
                                     specs_new; previous_specs=specs_old)
    @test state.mean[[1, 2, 5]] == [0.3, 0.4, 0.2]
    @test state.mean[3] == 0.4
    @test length(state.mean) == 5
    @test O.validate_cma_state(state, 5)["valid"]

    seed = Dict{String,Any}("curve" => Dict{String,Any}(
        "interval_values" => [0.2, 0.4, 0.8],
        "interval_times" => [1, 30, 31]))
    cfg = O.OptimizerConfig("seed", "out", 30, O.StageConfig[],
        Dict{String,Tuple{Float64,Float64}}(), Dict("curve" => (0.0, 1.0)),
        Dict{String,Dict{String,Any}}(), "monthly", Dict{String,Float64}(),
        Dict{String,Any}(), O.ObjectiveConfig(Dict{String,Float64}(), 1, 1.0, 0,
        "baseline", 0.0, 0.0), nothing, Dict{String,Vector{String}}(), nothing,
        O.PosteriorConfig(false, "", 1, 1, 1, .1, 1., 1., 1., 1., 0.))
    O.CURRENT_OPTIMIZER_CONFIG[] = cfg
    spec = [O.ParamSpec("curve.interval_values", :temporal, 3, 0., 1.)]
    @test O.initial_vector(seed, spec) == [0.4, 0.8, 0.8]
    candidate = O.vector_to_config(seed, spec, [0.1, 0.2, 0.3], 2)
    @test candidate["curve"]["interval_values"] == [0.1, 0.1, 0.2]
    @test candidate["stop_simulation_time"] == 60
    @test O.monthly_bucket(30, 30) == 1
    @test O.monthly_bucket(31, 30) == 2
    @test_throws ArgumentError O.validate_interval_times([1, 1])

    rng = MersenneTwister(17)
    snapshot = O.rng_snapshot(rng)
    expected = rand(rng, 4)
    restored = O.restore_rng(snapshot)
    @test rand(restored, 4) == expected

    # Transition integration must be explicit about provenance and RNG ownership.
    previous = Dict{String,Any}(
        "param_names" => ["curve[1]", "curve[2]"],
        "values" => [0.9, 0.4],
    )
    current = Dict{String,Any}(
        "param_names" => ["curve[1]", "curve[2]", "curve[3]"],
        "values" => [0.2, 0.4, 0.9],
    )
    report = O.transition_delta_report(previous, current; limit=0.1)
    @test report["policy_outcome"] == "reject"
    @test report["coordinates"][3]["class"] == "new_dimension"
    @test report["coordinates"][1]["provenance"] == "archive_transfer"

    reusable = Dict{String,Any}(
        "param_names" => ["curve[1]", "curve[2]"],
        "mean" => [0.2, 0.4],
        "sigma" => [0.1, 0.1],
        "covariance" => Matrix{Float64}(I, 2, 2),
        "p_c" => [0.0, 0.0],
        "p_sigma" => [0.0, 0.0],
    )
    seed2 = Dict{String,Any}("curve" => Dict{String,Any}(
        "interval_values" => [0.1, 0.1, 0.1],
        "interval_times" => [1, 31, 61],
    ))
    specs2 = [O.ParamSpec("curve.interval_values", :temporal, 3, 0.0, 1.0)]
    # Candidate transfer policy is enforced before any scorer/archive consumer.
    transfer_cfg = Dict{String,Any}("curve" => Dict{String,Any}("interval_values" => [0.9, 0.4, 0.9]))
    rejected = O.enforce_transition_policy(seed2, transfer_cfg, specs2,
        Dict{String,Any}("parameter_names" => ["curve.interval_values[1]", "curve.interval_values[2]", "curve.interval_values[3]"],
                         "evaluated_vector" => [0.1, 0.1, 0.1]);
        candidate_class="archive_transfer", limit=0.1, policy="reject")
    @test rejected["status"] == "rejected"
    @test rejected["report"]["policy_outcome"] == "reject"
    clipped = O.enforce_transition_policy(seed2, transfer_cfg, specs2,
        Dict{String,Any}("parameter_names" => ["curve.interval_values[1]", "curve.interval_values[2]", "curve.interval_values[3]"],
                         "evaluated_vector" => [0.1, 0.1, 0.1]);
        candidate_class="archive_transfer", limit=0.1, policy="clip")
    @test clipped["status"] == "accepted"
    @test clipped["report"]["policy_outcome"] == "clipped"
    @test clipped["config"]["curve"]["interval_values"] == [0.2, 0.2, 0.2]
    a = O.build_state_from_reusable(seed2, specs2, reusable; rng=MersenneTwister(8))
    b = O.build_state_from_reusable(seed2, specs2, reusable; rng=MersenneTwister(8))
    @test a.mean == b.mean

    posterior_state = Dict{String,Any}(
        "stage" => "posterior",
        "param_names" => ["curve.interval_values[1]", "curve.interval_values[2]",
                          "curve.interval_values[3]"],
        "mean" => [0.9, 0.4, 0.9],
        "sigma" => [0.1, 0.1, 0.1],
        "covariance" => Matrix{Float64}(I, 3, 3),
    )
    posterior_source = Dict{String,Any}(
        "parameter_names" => posterior_state["param_names"],
        "evaluated_vector" => [0.1, 0.1, 0.1],
        "candidate" => "archive-7",
        "stage" => "short",
        "fit_months" => 3,
    )
    rejected_posterior = O.enforce_posterior_reusable_state(
        seed2, specs2, posterior_state, posterior_source;
        active_months=3, limit=0.1, policy="reject",
    )
    @test rejected_posterior["status"] == "rejected"
    @test rejected_posterior["state"] === nothing
    @test rejected_posterior["report"]["policy_outcome"] == "reject"
    @test rejected_posterior["terminal_evidence"]["failure_class"] ==
          "posterior_transition_policy_reject"

    clipped_posterior = O.enforce_posterior_reusable_state(
        seed2, specs2, posterior_state, posterior_source;
        active_months=3, limit=0.1, policy="clip",
    )
    @test clipped_posterior["status"] == "accepted"
    @test clipped_posterior["state"]["mean"] == [0.2, 0.2, 0.2]
    @test clipped_posterior["report"]["policy_outcome"] == "clipped"
    @test clipped_posterior["state"]["transition_delta_report"] ===
          clipped_posterior["report"]
    @test clipped_posterior["report"]["source_provenance"]["archive_entry_id"] ==
          "archive-7"
    @test clipped_posterior["report"]["coordinates"][1]["raw_value"] == 0.9
    @test clipped_posterior["report"]["coordinates"][1]["effective_value"] == 0.2
    @test clipped_posterior["report"]["coordinates"][1]["class"] == "archive_transfer"
end
