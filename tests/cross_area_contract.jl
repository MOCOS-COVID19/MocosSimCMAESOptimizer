using Test
using JSON
using HDF5

const REPO = normpath(joinpath(@__DIR__, ".."))
const JULIA = get(ENV, "JULIA", "/Users/marcinbodych/Workspace/saxocov/julia-1.7.0/bin/julia")

@testset "fixture-only staged two-year pipeline" begin
    root = mktempdir()
    cfgdir = joinpath(root, "cfg")
    mkpath(cfgdir)
    h5open(joinpath(cfgdir, "population.jld2"), "w") do f
        write(f, "individuals_df", ones(Float64, 4, 2))
    end
    h5open(joinpath(cfgdir, "covimod.jld2"), "w") do f
        write(f, "age_thresholds", [0.0, 1.0, 2.0])
        write(f, "contact_mat", ones(Float64, 3, 3))
        write(f, "uses_genders", false)
    end
    h5open(joinpath(cfgdir, "events.jld2"), "w") do f
        write(f, "events", ones(Int, 2))
    end
    seed = Dict("population_path"=>"population.jld2",
        "transmission_probabilities"=>Dict("age_coupling_data_path"=>"covimod.jld2",
                                            "constant"=>0.1),
        "initial_conditions"=>Dict("immunization"=>Dict("immunity_events"=>"events.jld2")),
        "infection_modulation"=>Dict("function"=>"IntervalsModulations",
            "params"=>Dict("interval_times"=>[30, 60, 90, 120, 150, 180, 210, 240],
                           "interval_values"=>fill(0.2, 8))))
    open(joinpath(cfgdir, "seed.json"), "w") do io JSON.print(io, seed) end
    mkpath(joinpath(cfgdir, "gt"))
    write(joinpath(cfgdir, "gt", "daily.csv"), "day,value\n1,1\n2,2\n")
    cfg = Dict("seed_config"=>"seed.json", "output_dir"=>"unused",
        "monthly_days"=>30,
        "stages"=>[Dict("name"=>"fixture", "fit_months"=>1, "max_iterations"=>1,
                        "population_size"=>1, "sigma"=>0.1)],
        "scalar_bounds"=>Dict("transmission_probabilities.constant"=>[0.0, 1.0]),
        "temporal_bounds"=>Dict{String,Any}(),
        "objective"=>Dict("weights"=>Dict("daily_detections"=>1.0)),
        "validation"=>Dict("enabled"=>false), "age_population_weights"=>Dict("all"=>1.0),
        "gt_dir"=>"gt", "julia_bin"=>"fake-julia", "project_dir"=>".",
        "advanced_cli"=>"advanced_cli.jl")
    write(joinpath(cfgdir, "fake-julia"), "#!/bin/sh\nexit 0\n")
    write(joinpath(cfgdir, "advanced_cli.jl"), "# fixture boundary\n")
    chmod(joinpath(cfgdir, "fake-julia"), 0o755)
    config_path = joinpath(cfgdir, "config.json")
    open(config_path, "w") do io JSON.print(io, cfg) end
    batch = Dict("base_config"=>"config.json", "output_root"=>"fixture-run",
        "adapter_mode"=>"fixture", "target_months"=>24,
        "stages"=>[Dict("name"=>"stage_06m", "fit_months"=>6),
                   Dict("name"=>"stage_12m", "fit_months"=>12),
                   Dict("name"=>"stage_18m", "fit_months"=>18),
                   Dict("name"=>"stage_24m", "fit_months"=>24)])
    batch_path = joinpath(cfgdir, "batch.json")
    open(batch_path, "w") do io JSON.print(io, batch) end
    output = read(`$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl $batch_path`, String)
    summary = JSON.parse(output)
    @test summary["status"] == "fixture_complete"
    @test summary["target_months"] == 24
    @test summary["deferred"]["simulation"] == "DEFERRED"
    @test summary["deferred"]["multi_seed"] == "DEFERRED"
    @test summary["deferred"]["slurm"] == "DEFERRED"
    @test length(summary["stages"]) == 4
    @test [s["fit_months"] for s in summary["stages"]] == [6, 12, 18, 24]
    @test all(s["gate"]["status"] == "passed" for s in summary["stages"])
    @test all(isfile(joinpath(s["stage_root"], "preflight_manifest.json"))
              for s in summary["stages"])
    # Every extension must consume the canonical archive finalized by its
    # immediate predecessor, while current-stage survivor selection remains a
    # separate artifact.
    for i in 2:length(summary["stages"])
        previous = summary["stages"][i - 1]
        current = summary["stages"][i]
        transfer = JSON.parsefile(joinpath(current["stage_root"], "transfer_manifest.json"))
        @test transfer["source_archive_path"] == previous["archive_path"]
        @test transfer["source_stage"] == previous["name"]
        @test transfer["target_stage"] == current["name"]
        @test transfer["source_horizon_months"] == previous["fit_months"]
        @test transfer["admitted_ids"] == previous["archive_ids"]
        @test transfer["candidate_order"] == previous["archive_ids"]
        @test transfer["protected_transfer_slots"] == previous["archive_ids"]
        @test isempty(intersect(Set(transfer["protected_transfer_slots"]),
                                Set(transfer["immigrant_slots"])))
        @test JSON.parsefile(joinpath(current["stage_root"], "transfer_candidates.json")) ==
              JSON.parsefile(joinpath(previous["stage_root"], "survivor_archive.json"))
        @test current["archive_ids"] != transfer["admitted_ids"]
    end
    @test isfile(joinpath(summary["output_root"], "pipeline_summary.json"))
    @test !isdir(joinpath(summary["output_root"], "advanced_cli.jl"))
end
