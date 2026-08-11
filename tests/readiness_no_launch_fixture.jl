using Test
using JSON
using HDF5

const REPO = normpath(joinpath(@__DIR__, ".."))
const JULIA = get(ENV, "JULIA", "/Users/marcinbodych/Workspace/saxocov/julia-1.7.0/bin/julia")

function readiness_fixture()
    root = mktempdir()
    cfgdir = joinpath(root, "config")
    mkpath(cfgdir)

    function write_model(path; kind)
        h5open(path, "w") do f
            if kind == :population
                write(f, "individuals_df", ones(Float64, 4, 2))
            elseif kind == :covimod
                write(f, "age_thresholds", [0.0, 1.0, 2.0])
                write(f, "contact_mat", ones(Float64, 3, 3))
                write(f, "uses_genders", false)
            else
                write(f, "events", ones(Int, 2))
            end
        end
    end
    write_model(joinpath(cfgdir, "population.jld2"); kind=:population)
    write_model(joinpath(cfgdir, "covimod.jld2"); kind=:covimod)
    write_model(joinpath(cfgdir, "events.jld2"); kind=:events)

    seed = Dict{String,Any}(
        "population_path" => "population.jld2",
        "transmission_probabilities" => Dict(
            "age_coupling_data_path" => "covimod.jld2",
            "constant" => 0.1,
        ),
        "initial_conditions" => Dict(
            "immunization" => Dict("immunity_events" => "events.jld2"),
        ),
        "infection_modulation" => Dict(
            "function" => "IntervalsModulations",
            "params" => Dict("interval_times" => [30],
                             "interval_values" => [0.2]),
        ),
    )
    seed_path = joinpath(cfgdir, "seed.json")
    open(seed_path, "w") do io JSON.print(io, seed) end
    gt = joinpath(cfgdir, "gt")
    mkpath(gt)
    write(joinpath(gt, "daily.csv"), "day,value\n1,1\n2,2\n")

    sentinel = joinpath(root, "launched.txt")
    fake_julia = joinpath(cfgdir, "fake-julia.sh")
    fake_launcher = joinpath(cfgdir, "advanced_cli.jl")
    write(fake_julia, "#!/bin/sh\ntouch \"$sentinel\"\nexit 99\n")
    write(fake_launcher, "#!/bin/sh\ntouch \"$sentinel\"\nexit 99\n")
    chmod(fake_julia, 0o755)
    chmod(fake_launcher, 0o755)

    cfg = Dict{String,Any}(
        "seed_config" => "seed.json",
        "output_dir" => "fixture-output",
        "monthly_days" => 30,
        "stages" => [Dict("name" => "fixture", "fit_months" => 1,
                          "max_iterations" => 1, "population_size" => 1,
                          "sigma" => 0.1)],
        "scalar_bounds" => Dict("transmission_probabilities.constant" => [0.0, 1.0]),
        "temporal_bounds" => Dict{String,Any}(),
        "objective" => Dict("weights" => Dict("daily_detections" => 1.0)),
        "validation" => Dict("enabled" => false),
        "age_population_weights" => Dict("all" => 1.0),
        "gt_dir" => "gt",
        "julia_bin" => "fake-julia.sh",
        "project_dir" => ".",
        "advanced_cli" => "advanced_cli.jl",
    )
    config_path = joinpath(cfgdir, "config.json")
    open(config_path, "w") do io JSON.print(io, cfg) end
    return root, config_path, sentinel
end

@testset "fixture-only readiness never launches deferred execution" begin
    root, config_path, sentinel = readiness_fixture()
    output = read(`$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl --readiness $config_path`,
                  String)
    manifest = JSON.parse(output)
    readiness_root = manifest["readiness_root"]
    persisted = JSON.parsefile(joinpath(readiness_root, "preflight_manifest.json"))

    @test manifest["valid"] == true
    @test manifest["invocation"]["advanced_cli"] == false
    @test manifest["deferred"] == ["advanced_cli.jl", "validation_replicates", "Slurm"]
    @test persisted["readiness_root"] == readiness_root
    @test persisted["expected_artifacts"] == ["preflight_manifest.json",
                                               "no_simulation_invocations"]
    @test !isfile(sentinel)
    @test !isdir(joinpath(root, "config", "fixture-output"))
    @test startswith(readiness_root, root)
end
