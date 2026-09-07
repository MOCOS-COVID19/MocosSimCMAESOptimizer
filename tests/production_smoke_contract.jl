using Test
using JSON
using HDF5

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function hermetic_smoke_fixture()
    root = mktempdir()
    for (name, dataset) in (("population.jld2", "individuals_df"),
                            ("covimod.jld2", "contact_mat"),
                            ("events.jld2", "events"))
        h5open(joinpath(root, name), "w") do file
            write(file, dataset, ones(2, 2))
            if name == "covimod.jld2"
                write(file, "age_thresholds", [0.0, 1.0])
                write(file, "uses_genders", false)
            end
        end
    end
    seed = Dict(
        "population_path" => "population.jld2",
        "transmission_probabilities" => Dict(
            "constant" => 0.1, "age_coupling_data_path" => "covimod.jld2"),
        "initial_conditions" => Dict("immunization" =>
            Dict("immunity_events" => "events.jld2")))
    O.save_json(joinpath(root, "seed.json"), seed)

    gt = joinpath(root, "gt")
    mkpath(gt)
    for name in ("daily_age_total_detections.csv", "daily_hospitalizations.csv",
                 "daily_age_total_deaths.csv",
                 "sax-scholars-infections-normalized.csv")
        open(joinpath(gt, name), "w") do io
            println(io, "day,value")
            for day in 1:90
                println(io, "$day,1")
            end
        end
    end

    launcher = joinpath(root, "advanced_cli.jl")
    write(launcher, """
using HDF5
index = findfirst(==("--output-daily"), ARGS)
index === nothing && error("missing --output-daily")
h5open(ARGS[index + 1], "w") do file
    trajectory = create_group(file, "trajectory_1")
    for metric in ("daily_detections", "daily_deaths", "daily_hospitalizations")
        write(trajectory, metric, ones(90))
    end
end
""")
    config = Dict{String,Any}(
        "seed_config" => "seed.json", "output_dir" => "output",
        "monthly_days" => 30,
        "stages" => [Dict("name" => "production_smoke_3m", "fit_months" => 3,
                          "max_iterations" => 1, "population_size" => 2,
                          "sigma" => 0.1)],
        "scalar_bounds" => Dict("transmission_probabilities.constant" => [0.05, 0.2]),
        "temporal_bounds" => Dict{String,Any}(),
        "objective" => Dict("weights" => Dict(
            "daily_detections" => 1.0, "daily_deaths" => 1.0,
            "daily_hospitalizations" => 1.0, "weekly_control" => 0.0), "top_k" => 2,
            "finish_iter_delay" => 0),
        "posterior" => Dict("enabled" => false),
        "validation" => Dict("enabled" => false, "adapter_timeout_seconds" => 60),
        "gt_dir" => gt,
        "julia_bin" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "project_dir" => joinpath(@__DIR__, ".."),
        "advanced_cli" => launcher)
    config_path = joinpath(root, "smoke.json")
    O.save_json(config_path, config)
    return root, config_path
end

@testset "production daily output validation" begin
    root = mktempdir()
    valid_path = joinpath(root, "daily.jld2")
    h5open(valid_path, "w") do file
        trajectory = create_group(file, "trajectory_1")
        for metric in ("daily_detections", "daily_deaths", "daily_hospitalizations")
            write(trajectory, metric, [1.0, 2.0, 3.0])
        end
    end
    report = O.validate_simulation_jld2(valid_path; minimum_days=3)
    @test report["valid"]
    @test report["trajectory_count"] == 1
    @test report["identity"]["bytes"] > 0

    invalid_path = joinpath(root, "invalid.jld2")
    h5open(invalid_path, "w") do file
        trajectory = create_group(file, "trajectory_1")
        write(trajectory, "daily_detections", [1.0, NaN])
    end
    invalid = O.validate_simulation_jld2(invalid_path)
    @test !invalid["valid"]
    @test any(occursin("non-finite", error) for error in invalid["errors"])
    @test any(occursin("daily_deaths", error) for error in invalid["errors"])
end

@testset "hermetic production smoke exercises the Julia adapter" begin
    root, config = hermetic_smoke_fixture()
    manifest = O.run_production_smoke(config; use_slurm=false)
    @test manifest["status"] == "passed"
    @test manifest["execution_mode"] == "local"
    @test length(manifest["candidates"]) == 2
    @test all(candidate["status"]["status"] == "completed" for
              candidate in manifest["candidates"])
    @test all(candidate["output_validation"]["valid"] for
              candidate in manifest["candidates"])
    @test all(candidate["adapter_invocation"]["success"] for
              candidate in manifest["candidates"])
    @test isfile(joinpath(root, "output", "production_smoke_manifest.json"))
end

function parity_manifest(mode, input_hash="same"; trajectories=1)
    validation = Dict{String,Any}(
        "required_metrics" => ["daily_detections", "daily_deaths",
                               "daily_hospitalizations"],
        "trajectory_count" => trajectories,
        "trajectories" => [Dict("name" => "trajectory_1", "dimensions" =>
            Dict(metric => 90 for metric in ("daily_detections", "daily_deaths",
                                              "daily_hospitalizations")))])
    return Dict{String,Any}(
        "execution_mode" => mode, "optimizer_commit" => "optimizer",
        "launcher_commit" => "launcher",
        "input_identities" => Dict("config" => Dict("sha256" => input_hash)),
        "stage" => Dict("name" => "production_smoke_3m", "iterations" => 1,
                        "population_size" => 2),
        "candidates" => [Dict("output_validation" => deepcopy(validation)),
                         Dict("output_validation" => deepcopy(validation))])
end

@testset "local and Slurm parity contract" begin
    root = mktempdir()
    local_path, slurm_path = joinpath(root, "local.json"), joinpath(root, "slurm.json")
    O.safe_save_json(local_path, parity_manifest("local"))
    O.safe_save_json(slurm_path, parity_manifest("slurm"))
    @test O.compare_smoke_manifests(local_path, slurm_path)["status"] == "passed"
    O.safe_save_json(slurm_path, parity_manifest("slurm", "different"))
    failed = O.compare_smoke_manifests(local_path, slurm_path)
    @test failed["status"] == "failed"
    @test "input identities differ" in failed["contradictions"]
end

@testset "real production smoke when external inputs are supplied" begin
    required = ("JULIA_BIN", "MOCOSSIM_LAUNCHER_DIR", "MOCOSSIM_ADVANCED_CLI",
                "MOCOSSIM_SEED_CONFIG")
    available = all(haskey(ENV, name) && ispath(ENV[name]) for name in required)
    if !available
        @info "Skipping real smoke: external Saxony seed, JLD2 inputs, and launcher were not supplied"
        @test_skip available
    else
        output = mktempdir()
        config = joinpath(@__DIR__, "..", "optimizer_config.saxony.smoke.json")
        manifest = withenv("MOCOSSIM_SMOKE_OUTPUT" => output) do
            O.run_production_smoke(config; use_slurm=false)
        end
        @test manifest["status"] == "passed"
        @test length(manifest["candidates"]) == 2
        @test all(candidate["output_validation"]["valid"] for
                  candidate in manifest["candidates"])
    end
end
