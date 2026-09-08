using Test
using JSON
using HDF5
using SHA

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

const O = MocosSimCMAESOptimizer

function write_model_fixture(path::String; population_rows=4, covimod_dim=3,
                             contact_dim=covimod_dim, event_rows=2, missing_key=nothing)
    h5open(path, "w") do f
        missing_key == :population || write(f, "individuals_df", ones(Float64, population_rows, 2))
        if missing_key != :covimod
            write(f, "age_thresholds", collect(0.0:1.0:covimod_dim - 1))
            write(f, "contact_mat", ones(Float64, contact_dim, contact_dim))
            write(f, "uses_genders", false)
        end
        missing_key == :events || write(f, "events", ones(Int, event_rows))
    end
end

function model_schema_fixture(; missing_key=nothing, bad_gt=false,
                              stage_months=2, interval_times=[30, 60],
                              interval_values=[0.2, 0.3], population_rows=4,
                              covimod_dim=3, contact_dim=covimod_dim)
    root = mktempdir()
    cfgdir = joinpath(root, "config")
    mkpath(cfgdir)
    write_model_fixture(joinpath(cfgdir, "population.jld2");
                        population_rows=population_rows, missing_key=missing_key)
    write_model_fixture(joinpath(cfgdir, "covimod.jld2");
                        covimod_dim=covimod_dim, contact_dim=contact_dim,
                        missing_key=missing_key)
    write_model_fixture(joinpath(cfgdir, "events.jld2");
                        event_rows=2, missing_key=missing_key)
    seed = Dict{String,Any}(
        "population_path" => "population.jld2",
        "transmission_probabilities" => Dict(
            "age_coupling_data_path" => "covimod.jld2",
            "constant" => 0.1, "household" => 0.1, "school" => 0.2,
        ),
        "initial_conditions" => Dict(
            "immunization" => Dict("immunity_events" => "events.jld2"),
        ),
        "infection_modulation" => Dict("function" => "IntervalsModulations",
            "params" => Dict("interval_times" => interval_times,
                             "interval_values" => interval_values)),
    )
    seed_path = joinpath(cfgdir, "seed.json")
    open(seed_path, "w") do io JSON.print(io, seed) end
    gt = joinpath(cfgdir, "gt")
    mkpath(gt)
    if bad_gt
        write(joinpath(gt, "daily.csv"), "date,observed\n1,1\n1,2\n")
    else
        write(joinpath(gt, "daily.csv"), "day,value\n1,1\n2,2\n")
    end
    julia = joinpath(cfgdir, "julia")
    launcher = joinpath(cfgdir, "advanced_cli.jl")
    write(julia, "#!/bin/sh\n"); chmod(julia, 0o755)
    write(launcher, "# fixture\n")
    mkpath(joinpath(cfgdir, "launcher"))
    cfg = Dict{String,Any}(
        "seed_config" => "seed.json", "output_dir" => "out",
        "monthly_days" => 30,
        "stages" => [Dict("name"=>"short", "fit_months"=>stage_months,
                          "max_iterations"=>1, "population_size"=>1, "sigma"=>0.1)],
        "scalar_bounds" => Dict("transmission_probabilities.school" => [0.0, 1.0]),
        "temporal_bounds" => Dict("infection_modulation.params.interval_values" => [0.0, 1.0]),
        "objective" => Dict("weights"=>Dict("daily_detections"=>1.0)),
        "age_population_weights" => Dict("all"=>1.0),
        "gt_dir" => "gt", "julia_bin" => "julia",
        "project_dir" => "launcher", "advanced_cli" => "advanced_cli.jl",
    )
    path = joinpath(cfgdir, "config.json")
    open(path, "w") do io JSON.print(io, cfg) end
    return root, path, [joinpath(cfgdir, x) for x in ("population.jld2", "covimod.jld2", "events.jld2")]
end

@testset "configuration preflight is fail closed and path anchored" begin
    root = mktempdir()
    cfgdir = joinpath(root, "config")
    mkpath(cfgdir)
    seed = joinpath(cfgdir, "seed.json")
    open(seed, "w") do io
        JSON.print(io, Dict(
            "population_path" => "population.jld2",
            "transmission_probabilities" => Dict(
                "age_coupling_data_path" => "covimod.jld2",
                "constant" => 0.1,
            ),
            "immunity_events_path" => "events.jld2",
        ))
    end
    write_model_fixture(joinpath(cfgdir, "population.jld2"))
    write_model_fixture(joinpath(cfgdir, "covimod.jld2"))
    write_model_fixture(joinpath(cfgdir, "events.jld2"))
    gt = joinpath(cfgdir, "gt")
    mkpath(gt)
    write(joinpath(gt, "daily_detections.csv"), "day,value\n1,1\n2,2\n")
    write(joinpath(gt, "daily_deaths.csv"), "day,value\n1,0\n")
    julia = joinpath(cfgdir, "julia")
    launcher = joinpath(cfgdir, "advanced_cli.jl")
    project = joinpath(cfgdir, "launcher")
    write(julia, "#!/bin/sh\n")
    chmod(julia, 0o755)
    write(launcher, "# fixture\n")
    mkpath(project)
    cfg = Dict{String,Any}(
        "seed_config" => "seed.json", "output_dir" => "out",
        "monthly_days" => 30, "temporal_parameterization" => "monthly",
        "stages" => [Dict("name"=>"short", "fit_months"=>1,
                          "max_iterations"=>1, "population_size"=>1, "sigma"=>0.1)],
        "scalar_bounds" => Dict("transmission_probabilities.constant" => [0.0, 1.0]),
        "temporal_bounds" => Dict{String,Any}(),
        "objective" => Dict("weights"=>Dict("daily_detections"=>1.0),
                            "top_k"=>1, "min_completion_fraction"=>0.5,
                            "finish_iter_delay"=>0),
        "validation" => Dict("enabled"=>false, "holdout_days"=>1),
        "age_population_weights" => Dict("all"=>1.0),
        "gt_dir" => "gt", "julia_bin" => "julia",
        "project_dir" => "launcher", "advanced_cli" => "advanced_cli.jl",
    )
    path = joinpath(cfgdir, "config.json")
    open(path, "w") do io JSON.print(io, cfg) end
    manifest = O.preflight_config(path; readiness=true)
    @test manifest["valid"]
    @test manifest["effective_completion_threshold"] == 0.9
    @test manifest["paths"]["seed_config"] == seed
    @test manifest["paths"]["population"] == joinpath(cfgdir, "population.jld2")
    @test manifest["invocation"]["advanced_cli"] == false
    @test !isdir(joinpath(cfgdir, "out"))

    cfg["stages"][1]["population_size"] = 0
    bad = joinpath(cfgdir, "bad.json")
    open(bad, "w") do io JSON.print(io, cfg) end
    err = try O.preflight_config(bad) catch e; e end
    @test err isa ArgumentError
    @test occursin("stages[1].population_size", sprint(showerror, err))
    @test !isdir(joinpath(cfgdir, "out"))
end

@testset "model and ground-truth schemas are validated before launch" begin
    root, path, sources = model_schema_fixture()
    hashes = [bytes2hex(open(sha256, p)) for p in sources]
    manifest = O.preflight_config(path; readiness=true)
    @test manifest["valid"]
    @test manifest["model_inputs"]["population"]["required_keys"] == ["individuals_df"]
    @test manifest["model_inputs"]["covimod"]["contact_matrix_shape"] == [3, 3]
    @test manifest["model_inputs"]["immunity_events"]["event_count"] == 2
    @test manifest["ground_truth"]["daily.csv"]["day_policy"] == "positive_unique_strictly_increasing"
    @test [bytes2hex(open(sha256, p)) for p in sources] == hashes
    @test !isdir(joinpath(dirname(path), "out"))

    # The raw Saxony scholars source has its own date/semicolon schema. It is
    # validated as provenance and checked against the normalized input.
    gt = joinpath(dirname(path), "gt")
    write(joinpath(gt, "sax-scholars-infections.csv"),
          "calendar_week_date;students_infected_weekly\n2020-11-12;198\n2020-11-18;145\n")
    write(joinpath(gt, "sax-scholars-infections-normalized.csv"),
          "day,value\n72,198\n78,145\n")
    manifest_with_source = O.preflight_config(path; readiness=true)
    @test manifest_with_source["valid"]
    source_entry = manifest_with_source["ground_truth"]["sax-scholars-infections.csv"]
    @test source_entry["schema"] == "calendar_date_semicolon_source"
    @test source_entry["normalized_values_match"]

    write(joinpath(gt, "sax-scholars-infections-normalized.csv"),
          "day,value\n72,999\n78,145\n")
    err = try O.preflight_config(path; readiness=true) catch e; e end
    @test err isa ArgumentError
    @test occursin("values differ", sprint(showerror, err))

    # IntervalsModulations also supports N interval values separated by N - 1
    # boundary times, which is the schema used by the production seed.
    _, boundary_path, _ = model_schema_fixture(interval_times=[30, 60],
                                                interval_values=[0.2, 0.3, 0.4])
    @test O.preflight_config(boundary_path; readiness=true)["valid"]

    _, missing_path, _ = model_schema_fixture(missing_key=:covimod)
    err = try O.preflight_config(missing_path) catch e; e end
    @test err isa ArgumentError
    @test occursin("covimod", lowercase(sprint(showerror, err)))

    _, dimension_path, _ = model_schema_fixture(covimod_dim=4, contact_dim=3)
    err = try O.preflight_config(dimension_path) catch e; e end
    @test err isa ArgumentError
    @test occursin("dimension", lowercase(sprint(showerror, err)))

    _, temporal_path, _ = model_schema_fixture(interval_values=[0.2])
    err = try O.preflight_config(temporal_path) catch e; e end
    @test err isa ArgumentError
    @test occursin("interval", lowercase(sprint(showerror, err)))

    _, gt_path, _ = model_schema_fixture(bad_gt=true)
    err = try O.preflight_config(gt_path) catch e; e end
    @test err isa ArgumentError
    @test occursin("header", lowercase(sprint(showerror, err)))
end

@testset "adapter failures are explicit and roots are isolated" begin
    root = mktempdir()
    a = O.adapter_failure("missing_output"; command=["julia", "advanced_cli.jl"], exit_code=0)
    @test a["status"] == "failed"
    @test a["failure_class"] == "missing_output"
    @test isinf(a["penalty"])
    first_root = O.create_candidate_root(joinpath(root, "runs"), "cand_1")
    second_root = O.create_candidate_root(joinpath(root, "runs"), "cand_1")
    @test first_root != second_root
    @test isdir(first_root) && isdir(second_root)
end

@testset "configuration scrutiny gaps are closed" begin
    root, path, _ = model_schema_fixture(stage_months=2, interval_times=[30, 60])
    cfg = O.load_json(path)
    push!(cfg["stages"], Dict("name"=>"long", "fit_months"=>3,
                              "max_iterations"=>1, "population_size"=>1, "sigma"=>0.1))
    open(path, "w") do io JSON.print(io, cfg) end
    err = try O.preflight_config(path) catch e; e end
    @test err isa ArgumentError
    @test occursin("requested horizon 90", sprint(showerror, err))

    # Runtime config must retain normalized nested paths without rewriting the
    # source seed file.
    root2, path2, _ = model_schema_fixture()
    runtime = O.load_config(path2)
    @test hasproperty(runtime, :runtime_seed)
    @test runtime.runtime_seed["population_path"] ==
          joinpath(dirname(runtime.seed_config), "population.jld2")
    @test runtime.runtime_seed["transmission_probabilities"]["age_coupling_data_path"] ==
          joinpath(dirname(runtime.seed_config), "covimod.jld2")

    # Day and value columns are identified by name, not their physical order.
    gt = joinpath(dirname(path2), "gt")
    write(joinpath(gt, "daily_age_total_detections.csv"),
          "value,day,extra\n5,1,y\n10,2,x\n")
    loaded = O.load_gt_series(gt)
    @test loaded["daily_age_total_detections"][1:2] == [5.0, 10.0]

    # Optional status is based on actual files and validation, not a fixed
    # all-absent declaration.
    manifest = O.preflight_config(path2; readiness=true)["ground_truth"]
    @test manifest["optional_fields"]["daily_age_total_detections"]["present"]
    @test manifest["optional_fields"]["daily_age_total_detections"]["status"] == "valid"
    @test !manifest["optional_fields"]["household_infections"]["present"]
end

@testset "malformed optional ground truth is reported without aborting preflight" begin
    root, path, sources = model_schema_fixture()
    gt = joinpath(dirname(path), "gt")
    optional_path = joinpath(gt, "daily_age_00_04_detections.csv")
    write(optional_path, "observed,day\nnot-a-number,1\n")
    optional_hash = bytes2hex(open(sha256, optional_path))

    manifest = O.preflight_config(path; readiness=true)
    entry = manifest["ground_truth"]["optional_fields"]["daily_age_00_04_detections"]
    @test manifest["valid"]
    @test entry["present"] === true
    @test entry["validation_status"] == "invalid"
    @test entry["status"] == "invalid"
    @test entry["error"]["code"] == "invalid_value"
    @test entry["error"]["message"] ==
          "ground_truth.daily_age_00_04_detections.csv invalid value at row 2"
    @test entry["sha256"] == optional_hash

    required_path = joinpath(gt, "required_malformed.csv")
    write(required_path, "day,value\n1,not-a-number\n")
    err = try O.preflight_config(path; readiness=true) catch e; e end
    @test err isa ArgumentError
    @test occursin("required_malformed.csv invalid value", sprint(showerror, err))
end
