#!/usr/bin/env julia

using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

function resolve_path(path::String, base::String)
    return isabspath(path) ? path : normpath(joinpath(base, path))
end

function phase_config(base_config, phase, output_dir, seed_config, batch_base)
    cfg = deepcopy(base_config)
    cfg["output_dir"] = output_dir
    cfg["seed_config"] = seed_config
    cfg["stages"] = [Dict(
        "name" => String(phase["name"]),
        "fit_months" => Int(phase["fit_months"]),
        "max_iterations" => Int(phase["max_iterations"]),
        "population_size" => Int(phase["population_size"]),
        "sigma" => Float64(phase["sigma"]),
    )]
    early_stop = get(batch_base, "early_stop", Dict{String,Any}())
    objective = deepcopy(cfg["objective"])
    objective["min_completion_fraction"] = get(
        early_stop, "min_completion_fraction",
        get(objective, "min_completion_fraction", 1.0),
    )
    objective["finish_iter_delay"] = get(
        early_stop, "finish_iter_delay",
        get(objective, "finish_iter_delay", 30),
    )
    cfg["objective"] = objective
    return cfg
end

function write_phase_config(path, cfg)
    open(path, "w") do io
        JSON.print(io, cfg, 2)
    end
end

function run_pipeline(batch_path::String)
    batch_dir = dirname(abspath(batch_path))
    batch = JSON.parsefile(batch_path)
    base_path = resolve_path(String(batch["base_config"]), batch_dir)
    base_config = JSON.parsefile(base_path)
    output_root = resolve_path(String(batch["output_root"]), batch_dir)
    mkpath(output_root)
    use_slurm = Bool(get(batch, "use_slurm", true))

    short_phase = batch["short"]
    short_output = joinpath(output_root, String(short_phase["name"]))
    short_seed = resolve_path(
        String(get(short_phase, "seed_config", base_config["seed_config"])),
        batch_dir,
    )
    short_cfg = phase_config(base_config, short_phase, short_output, short_seed, batch)
    short_cfg_path = joinpath(output_root, "short_optimizer_config.json")
    write_phase_config(short_cfg_path, short_cfg)

    previous_posterior = get(batch, "initial_posterior", nothing)
    if previous_posterior !== nothing
        posterior_path = resolve_path(String(previous_posterior), batch_dir)
        reusable = posterior_reusable_state(posterior_path)
        safe_save_json(
            joinpath(short_output, "full_reusable_state.json"),
            reusable;
            label="initial_posterior_reusable_state",
        )
    end

    short_result = nothing
    if !Bool(get(batch, "skip_short", false))
        short_result = run_optimizer(short_cfg_path; use_slurm=use_slurm)
    elseif previous_posterior === nothing
        error("skip_short=true requires initial_posterior")
    end

    short_posterior = joinpath(
        short_output,
        "real_sims",
        String(short_phase["name"]),
        "posterior_samples.json",
    )
    isfile(short_posterior) || error("Short CMA-ES did not produce $short_posterior")
    short_best = joinpath(short_output, "final_best_candidate.json")
    isfile(short_best) || error("Short CMA-ES did not produce $short_best")

    long_phase = batch["long"]
    long_output = joinpath(output_root, String(long_phase["name"]))
    long_cfg_path = joinpath(output_root, "long_optimizer_config.json")
    long_seed = short_best
    long_cfg = phase_config(base_config, long_phase, long_output, long_seed, batch)
    write_phase_config(long_cfg_path, long_cfg)
    long_reusable = posterior_reusable_state(short_posterior)
    mkpath(long_output)
    safe_save_json(
        joinpath(long_output, "full_reusable_state.json"),
        long_reusable;
        label="short_posterior_reusable_state",
    )
    long_result = run_optimizer(long_cfg_path; use_slurm=use_slurm)

    summary = Dict(
        "batch_config" => abspath(batch_path),
        "short_config" => short_cfg_path,
        "short_output" => short_output,
        "short_posterior" => short_posterior,
        "long_config" => long_cfg_path,
        "long_output" => long_output,
        "long_posterior" => joinpath(long_output, "posterior_samples.json"),
        "short_result" => short_result,
        "long_result" => long_result,
    )
    safe_save_json(joinpath(output_root, "pipeline_summary.json"), summary; label="pipeline_summary")
    return summary
end

length(ARGS) == 1 || error("Usage: julia scripts/run_pipeline.jl pipeline.json")
println(JSON.json(run_pipeline(ARGS[1])))
