using Test
using JSON
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer
const ROOT = normpath(joinpath(@__DIR__, ".."))

@testset "corrected staged pilot configuration" begin
    raw = JSON.parsefile(joinpath(ROOT, "optimizer_config.saxony.corrected.json"))
    @test [s["fit_months"] for s in raw["stages"]] == [6, 9, 12]
    @test [s["max_iterations"] for s in raw["stages"]] == [10, 15, 20]
    @test raw["validation"]["active_temporal_tail_months"] == 3
    @test raw["validation"]["rank_on_validation"] == true
    @test raw["validation"]["plateau_patience"] == 3
    @test raw["validation"]["require_finite_validation_replicates"] == 3
    @test raw["validation"]["selection_replicate_seeds"] == [43, 44]
    @test raw["validation"]["selection_standard_error_penalty"] == 1.0
    @test raw["validation"]["current_quality_band"]["threshold"] < 1_000_000
end

@testset "two-phase calibration configuration" begin
    p1 = JSON.parsefile(joinpath(ROOT, "optimizer_config.saxony.phase1-scalars.json"))
    p2 = JSON.parsefile(joinpath(ROOT, "optimizer_config.saxony.phase2-vectors.json"))
    @test sort(p1["stage_freeze"]["phase1_scalar_6m"]) == sort(collect(keys(p1["temporal_bounds"])))
    @test sort(p2["stage_freeze"]["phase2_vector_6m"]) == sort(collect(keys(p2["scalar_bounds"])))
    @test p2["seed_config"] == "./runs/saxony-corrected-phase1-scalars/final_best_candidate.json"
    @test p1["validation"]["selection_objective_weights"] == p2["validation"]["selection_objective_weights"]
    @test isempty(p1["validation"]["selection_replicate_seeds"])
    @test length(p1["validation"]["validation_metric_weights"]) == 14
    @test sum(values(p1["validation"]["validation_metric_weights"])) ≈ 1.0
end

@testset "tail-only temporal coordinates preserve prefix" begin
    cfg = withenv(
        "MOCOSSIM_SEED_CONFIG" => joinpath(ROOT, "seed", "config2.json"),
        "JULIA_BIN" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "MOCOSSIM_LAUNCHER_DIR" => ROOT,
        "MOCOSSIM_ADVANCED_CLI" => joinpath(ROOT, "run_optimizer.jl"),
    ) do
        O.load_config(joinpath(ROOT, "optimizer_config.saxony.corrected.json"))
    end
    seed = cfg.runtime_seed
    specs = O.build_specs(seed, cfg)
    stage = cfg.stages[1]
    active = O.stage_specs(seed, specs, cfg, stage)
    @test length(O.coordinate_names(active)) == 12
    @test count(s -> s.kind == :scalar, active) == 3
    @test all(s.offset == 4 && s.length == 3 for s in active if s.kind == :temporal)
    x = O.initial_vector(seed, active)
    x[4:end] .= 0.321
    candidate = O.vector_to_config(seed, active, x, 6)
    for spec in active
        spec.kind == :temporal || continue
        before = O.get_nested(seed, spec.name)
        after = O.get_nested(candidate, spec.name)
        @test after[1:3] == before[1:3]
        @test after[4:6] == fill(0.321, 3)
    end
end

@testset "plateau requires three completed no-improvement generations" begin
    rows = [Dict("best_score" => 1.0, "median_score" => 2.0) for _ in 1:6]
    validation = Dict{String,Any}("plateau_patience" => 3,
        "plateau_min_iterations" => 6, "plateau_relative_tolerance" => 0.01)
    @test !O.plateau_reached(rows[1:5], validation)
    @test O.plateau_reached(rows, validation)
    rows[end]["best_score"] = 0.8
    @test !O.plateau_reached(rows, validation)
end
