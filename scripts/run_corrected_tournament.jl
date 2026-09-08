using JSON

const ROOT = normpath(joinpath(@__DIR__, ".."))
length(ARGS) >= 1 || error("usage: julia scripts/run_corrected_tournament.jl BASE_CONFIG [--slurm] [--prepare-only]")
base_path = abspath(ARGS[1])
slurm = "--slurm" in ARGS
prepare_only = "--prepare-only" in ARGS
base = JSON.parsefile(base_path)

policies = [
    ("baseline", "baseline", 42),
    ("temporal_escape", "temporal_escape", 42),
    # A restart is independent of baseline and deliberately uses a different
    # optimizer stream while retaining the same simulator/validation seeds.
    ("restart", "baseline", 1042),
]
configs = String[]
for (label, policy, optimizer_seed) in policies
    cfg = deepcopy(base)
    cfg["objective"]["search_policy"] = policy
    cfg["validation"]["optimizer_seed"] = optimizer_seed
    output_dir = joinpath(ROOT, "runs", "saxony-corrected-tournament", label)
    cfg["output_dir"] = output_dir
    ispath(output_dir) && error("refusing to reuse state in $output_dir; move or remove it explicitly")
    path = joinpath(dirname(base_path), "generated.corrected.$label.json")
    open(path, "w") do io
        JSON.print(io, cfg, 2)
    end
    push!(configs, path)
end

println("Prepared independent configs:\n", join(configs, "\n"))
prepare_only && exit(0)
for path in configs
    cmd = `$(Base.julia_cmd()) --project=$ROOT $(joinpath(ROOT, "run_optimizer.jl")) $path`
    slurm && (cmd = `$cmd --slurm`)
    run(cmd)
end
