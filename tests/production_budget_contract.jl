using Test
using JSON
using HDF5

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

const REPO = normpath(joinpath(@__DIR__, ".."))
const CONFIG = joinpath(REPO, "optimizer_config.saxony.12m-pilot.json")

@testset "twelve-month Saxony production pilot" begin
    config = JSON.parsefile(CONFIG)
    stages = config["stages"]
    @test [stage["fit_months"] for stage in stages] == [3, 6, 9, 12]
    @test [stage["max_iterations"] for stage in stages] == [4, 4, 4, 5]
    @test [stage["population_size"] for stage in stages] == [16, 16, 24, 24]

    candidate_budget = sum(stage["max_iterations"] * stage["population_size"]
                           for stage in stages)
    @test candidate_budget == 344
    @test candidate_budget <= floor(Int, 0.10 * 6384)
    @test config["output_dir"] == "./runs/saxony-12m-pilot"
    @test all(stage["sigma"] == 0.12 for stage in stages)
    @test config["objective"]["min_completion_fraction"] == 1.0
    @test config["objective"]["finish_iter_delay"] == 0
    @test config["posterior"]["enabled"] == false
    @test config["validation"]["adapter_timeout_seconds"] == 3300
    @test config["validation"]["iteration_timeout_seconds"] == 14400
    @test config["validation"]["current_minimum_archive_size"] == 5
    @test config["validation"]["likelihood_metrics"] ==
        ["daily_detections", "daily_deaths"]
    @test config["validation"]["likelihood_dispersions"] ==
        Dict("daily_detections" => 25.0, "daily_deaths" => 10.0)
    @test all(entry["mode"] == "normalize_to_bounds" for
              entry in values(config["scalar_preprocessing"]))

    wrapper = read(joinpath(REPO, "scripts", "run_cmaes.slurm"), String)
    @test occursin(r"CONFIG=\"\$\{1:-optimizer_config\.saxony\.12m-pilot\.json\}\"",
                   wrapper)
    @test occursin("run_optimizer.jl --preflight \"\$CONFIG\"", wrapper)
    @test occursin("run_optimizer.jl --slurm \"\$CONFIG\"", wrapper)
    @test occursin("#SBATCH -c 1", wrapper)

    optimizer_source = read(joinpath(REPO, "src", "MocosSimCMAESOptimizer.jl"), String)
    @test occursin("-t 01:15:00", optimizer_source)

    array_helper = read(joinpath(REPO, "scripts", "score_candidates.sh"), String)
    @test !occursin("Pkg.instantiate", array_helper)
    @test !occursin("uv pip install", array_helper)
    @test occursin("MOCOSSIM_PLOT_CANDIDATES", array_helper)
    @test occursin("Adapter succeeded but did not write", array_helper)
end

@testset "joint likelihood uses only declared metrics" begin
    root = mktempdir()
    daily = joinpath(root, "daily.jld2")
    h5open(daily, "w") do file
        trajectory = create_group(file, "trajectory_1")
        for metric in ("daily_detections", "daily_deaths",
                       "daily_hospitalizations", "daily_student_detections")
            write(trajectory, metric, ones(14))
        end
    end
    ground_truth = Dict(metric => fill(1.0, 14) for metric in
        ("daily_detections", "daily_deaths", "daily_hospitalizations",
         "daily_student_detections"))
    selected = ["daily_detections", "daily_deaths"]
    payload = O.vector_likelihood_payload(daily, ground_truth, 14;
        family="negative_binomial_weekly", metric_names=selected,
        dispersions=Dict("daily_detections" => 25.0, "daily_deaths" => 10.0))
    @test Set(row["metric"] for row in payload["dimensions"]) == Set(selected)
    @test length(payload["dimensions"]) == 4
    @test Set(row["dispersion"] for row in payload["dimensions"]
              if row["metric"] == "daily_detections") == Set([25.0])
    @test Set(row["dispersion"] for row in payload["dimensions"]
              if row["metric"] == "daily_deaths") == Set([10.0])
    @test_throws ArgumentError O.vector_likelihood_payload(
        daily, ground_truth, 14; family="negative_binomial_weekly",
        metric_names=["misspelled_metric"])
    @test_throws ArgumentError O.vector_likelihood_payload(
        daily, ground_truth, 14; family="negative_binomial_weekly",
        metric_names=["daily_deaths"], dispersions=Dict("daily_deaths" => 0.0))
end

@testset "completion threshold is loaded from config" begin
    root = mktempdir()
    seed = Dict(
        "transmission_probabilities" => Dict(
            "school" => 0.1, "class" => 0.2, "age_coupling_param" => 0.6),
        "infection_modulation" => Dict("params" => Dict(
            "interval_times" => collect(30:30:360), "interval_values" => fill(0.5, 12))),
        "mild_detection_modulation" => Dict("params" => Dict(
            "interval_times" => collect(30:30:360), "interval_values" => fill(0.5, 12))),
        "tracing_modulation" => Dict("params" => Dict(
            "interval_times" => collect(30:30:360), "interval_values" => fill(0.5, 12))))
    seed_path = joinpath(root, "seed.json")
    O.save_json(seed_path, seed)
    launcher = joinpath(root, "advanced_cli.jl")
    touch(launcher)
    loaded = withenv(
        "JULIA_BIN" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "MOCOSSIM_LAUNCHER_DIR" => root,
        "MOCOSSIM_ADVANCED_CLI" => launcher,
        "MOCOSSIM_SEED_CONFIG" => seed_path,
    ) do
        O.load_config(CONFIG)
    end
    @test loaded.objective.min_completion_fraction == 1.0
end
