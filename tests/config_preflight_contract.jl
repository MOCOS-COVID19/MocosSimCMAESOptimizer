using Test
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

const O = MocosSimCMAESOptimizer

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
            ),
            "immunity_events_path" => "events.jld2",
        ))
    end
    for name in ("population.jld2", "covimod.jld2", "events.jld2")
        write(joinpath(cfgdir, name), "fixture")
    end
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
        "scalar_bounds" => Dict("x" => [0.0, 1.0]),
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
