#!/usr/bin/env julia

using JSON
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer
const O = MocosSimCMAESOptimizer

function usage()
    println(stderr, "usage:")
    println(stderr, "  run_production_smoke.jl [--slurm] [CONFIG]")
    println(stderr, "  run_production_smoke.jl --compare LOCAL_MANIFEST SLURM_MANIFEST")
end

if "--compare" in ARGS
    index = findfirst(==("--compare"), ARGS)
    length(ARGS) >= index + 2 || (usage(); exit(2))
    result = O.compare_smoke_manifests(ARGS[index + 1], ARGS[index + 2])
    println(JSON.json(result))
    result["status"] == "passed" || exit(1)
else
    use_slurm = "--slurm" in ARGS
    positional = filter(!=("--slurm"), ARGS)
    config = isempty(positional) ?
        joinpath(@__DIR__, "..", "optimizer_config.saxony.smoke.json") : positional[1]
    result = O.run_production_smoke(config; use_slurm=use_slurm)
    println(JSON.json(result))
end
