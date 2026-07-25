#!/usr/bin/env julia

using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

function option(name, default)
    prefix = "--$(name)="
    for arg in ARGS
        startswith(arg, prefix) && return split(arg, "=", limit=2)[2]
    end
    return default
end

archive = length(ARGS) >= 1 && !startswith(ARGS[1], "--") ?
    ARGS[1] :
    error("Usage: julia scripts/run_nuts.jl ARCHIVE.jsonl|STAGE_DIR [--state=...] [--output=...]")
state = option("state", joinpath(dirname(archive), "cma_sampling_state.json"))
output = option("output", joinpath(isdir(archive) ? archive : dirname(archive), "posterior_samples.json"))
draws = parse(Int, option("draws", "1000"))
warmup = parse(Int, option("warmup", "500"))
max_depth = parse(Int, option("max-depth", "8"))
step_size = parse(Float64, option("step-size", "0.05"))
temperature = parse(Float64, option("temperature", "1.0"))
seed = parse(Int, option("seed", "42"))

result = isdir(archive) ?
    run_nuts_from_stage(
        archive;
        output_path=output,
        draws=draws,
        warmup=warmup,
        max_depth=max_depth,
        step_size=step_size,
        temperature=temperature,
        seed=seed,
    ) :
    run_nuts_from_archive(
        archive;
        state_path=state,
        output_path=output,
        draws=draws,
        warmup=warmup,
        max_depth=max_depth,
        step_size=step_size,
        temperature=temperature,
        seed=seed,
    )
println(JSON.json(Dict("posterior_samples" => result)))
