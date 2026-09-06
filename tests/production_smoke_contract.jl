using Test
using JSON
using HDF5

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

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
