module MocosSimCMAESOptimizer

using JSON
using LinearAlgebra
using Random
using Statistics
using Dates
using HDF5
using Printf

const MANAGER_ROOT = abspath(joinpath(@__DIR__, ".."))

export main, run_optimizer

struct ExternalSimConfig
    gt_dir::String
    julia_bin::String
    project_dir::String
    advanced_cli::String
end

struct StageConfig
    name::String
    fit_months::Int
    max_iterations::Int
    population_size::Int
    sigma::Float64
end

struct ObjectiveConfig
    weights::Dict{String,Float64}
    recent_days::Int
    early_reject_multiplier::Float64
end

struct OptimizerConfig
    seed_config::String
    output_dir::String
    monthly_days::Int
    stages::Vector{StageConfig}
    scalar_bounds::Dict{String,Tuple{Float64,Float64}}
    temporal_bounds::Dict{String,Tuple{Float64,Float64}}
    objective::ObjectiveConfig
    external_sim::Union{Nothing,ExternalSimConfig}
    stage_freeze::Dict{String,Vector{String}}
end

struct ParamSpec
    name::String
    kind::Symbol
    length::Int
    lower::Float64
    upper::Float64
end

struct CMAState
    mean::Vector{Float64}
    sigma::Float64
    covariance::Matrix{Float64}
end

function normalize_json(value)
    if value isa AbstractDict
        return Dict(k => normalize_json(v) for (k, v) in value)
    elseif value isa AbstractVector
        return map(normalize_json, value)
    else
        return value
    end
end

function load_json(path::String)
    open(path, "r") do io
        raw = JSON.parse(IOBuffer(read(io, String)))
        return normalize_json(raw)
    end
end

function save_json(path::String, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, value, 2)
    end
end

function load_config(path::String)
    raw = load_json(path)
    stages = [StageConfig(s["name"], s["fit_months"], s["max_iterations"], s["population_size"], float(s["sigma"])) for s in raw["stages"]]
    scalar_bounds = Dict(k => (float(v[1]), float(v[2])) for (k, v) in raw["scalar_bounds"])
    temporal_bounds = Dict(k => (float(v[1]), float(v[2])) for (k, v) in raw["temporal_bounds"])
    objective = ObjectiveConfig(Dict(k => float(v) for (k, v) in raw["objective"]["weights"]), Int(raw["objective"]["recent_days"]), float(raw["objective"]["early_reject_multiplier"]))
    external_sim = haskey(raw, "gt_dir") && haskey(raw, "julia_bin") && haskey(raw, "project_dir") && haskey(raw, "advanced_cli") ?
        ExternalSimConfig(String(raw["gt_dir"]), String(raw["julia_bin"]), String(raw["project_dir"]), String(raw["advanced_cli"])) :
        nothing
    stage_freeze = Dict{String,Vector{String}}()
    if haskey(raw, "stage_freeze")
        for (k, v) in raw["stage_freeze"]
            stage_freeze[String(k)] = [String(x) for x in v]
        end
    end
    return OptimizerConfig(raw["seed_config"], raw["output_dir"], Int(raw["monthly_days"]), stages, scalar_bounds, temporal_bounds, objective, external_sim, stage_freeze)
end

function get_nested(config::Dict{String,Any}, path::String)
    node = config
    parts = split(path, ".")
    for (i, part) in enumerate(parts)
        if i == length(parts)
            return node[part]
        end
        node = node[part]
    end
    return nothing
end

function set_nested!(config::Dict{String,Any}, path::String, value)
    node = config
    parts = split(path, ".")
    for part in parts[1:end-1]
        node = node[part]
    end
    node[parts[end]] = value
    return config
end

function build_specs(seed::Dict{String,Any}, cfg::OptimizerConfig)
    specs = ParamSpec[]
    for (name, (lo, hi)) in sort(collect(cfg.scalar_bounds), by=first)
        push!(specs, ParamSpec(name, :scalar, 1, lo, hi))
    end
    for (name, (lo, hi)) in sort(collect(cfg.temporal_bounds), by=first)
        arr = get_nested(seed, name)
        push!(specs, ParamSpec(name, :temporal, length(arr), lo, hi))
    end
    return specs
end

function stage_specs(specs::Vector{ParamSpec}, cfg::OptimizerConfig, stage::StageConfig)
    freeze = get(cfg.stage_freeze, stage.name, String[])
    isempty(freeze) && return specs
    return [spec for spec in specs if !(spec.name in freeze)]
end

function update_stage_freeze!(cfg::OptimizerConfig, stage::StageConfig, history::Vector{Any}, specs::Vector{ParamSpec}; min_calls::Int=5)
    counts = Dict{String,Int}(spec.name => 0 for spec in specs)
    for rec in history
        rec["stage"] == stage.name || continue
        score = get(rec, "score", Inf)
        isfinite(Float64(score)) || continue
        for spec in specs
            counts[spec.name] += 1
        end
    end
    movable = [name for (name, c) in counts if c >= min_calls]
    if length(movable) < 5
        freeze = String[]
    else
        freeze = [name for (name, c) in counts if c < min_calls]
    end
    cfg.stage_freeze[stage.name] = freeze
    return freeze
end

function initial_vector(seed::Dict{String,Any}, specs::Vector{ParamSpec})
    values = Float64[]
    for spec in specs
        current = get_nested(seed, spec.name)
        if spec.kind == :scalar
            push!(values, float(current))
        else
            append!(values, map(float, current))
        end
    end
    return values
end

function clip!(x::Vector{Float64}, specs::Vector{ParamSpec})
    idx = 1
    for spec in specs
        for _ in 1:spec.length
            x[idx] = clamp(x[idx], spec.lower, spec.upper)
            idx += 1
        end
    end
    return x
end

function vector_to_config(seed::Dict{String,Any}, specs::Vector{ParamSpec}, x::Vector{Float64}, active_months::Int)
    cfg = deepcopy(seed)
    idx = 1
    for spec in specs
        if spec.kind == :scalar
            set_nested!(cfg, spec.name, x[idx])
            idx += 1
        else
            current = map(float, get_nested(cfg, spec.name))
            active = min(active_months, spec.length)
            for i in 1:active
                current[i] = x[idx]
                idx += 1
            end
            for _ in active+1:spec.length
                idx += 1
            end
            set_nested!(cfg, spec.name, current)
        end
    end
    # Align simulation horizon with active months (e.g., 90d for first stage, 120d next, etc.)
    set_nested!(cfg, "stop_simulation_time", active_months * 30)
    return cfg
end

function inject_frozen!(cfg_out::Dict{String,Any}, seed::Dict{String,Any}, specs::Vector{ParamSpec}, frozen_names::Vector{String})
    for spec in specs
        spec.name in frozen_names || continue
        set_nested!(cfg_out, spec.name, get_nested(seed, spec.name))
    end
end

function rmse(a::Vector{Float64}, b::Vector{Float64})
    n = min(length(a), length(b))
    n == 0 && return Inf
    return sqrt(sum((a[i] - b[i])^2 for i in 1:n) / n)
end

function normalize_rmse(v::Float64, days::Int)
    isfinite(v) || return v
    return v / sqrt(max(days, 1))
end

function recent_weighted_rmse(a::Vector{Float64}, b::Vector{Float64}, recent_days::Int)
    n = min(length(a), length(b))
    n == 0 && return Inf
    start_idx = max(1, n - recent_days + 1)
    weights = collect(1.0:(n - start_idx + 1))
    err = 0.0
    denom = 0.0
    for (j, i) in enumerate(start_idx:n)
        w = weights[j]
        err += w * (a[i] - b[i])^2
        denom += w
    end
    return sqrt(err / max(denom, 1e-9))
end

function finite_diff_slope(series::Vector{Float64})
    length(series) < 2 && return zeros(Float64, 0)
    return [series[i+1] - series[i] for i in 1:length(series)-1]
end

function synthetic_simulation(config::Dict{String,Any}, days::Int)
    infection = map(float, config["infection_modulation"]["params"]["interval_values"])
    detection = map(float, config["mild_detection_modulation"]["params"]["interval_values"])
    tracing = map(float, config["tracing_modulation"]["params"]["interval_values"])
    household = float(config["transmission_probabilities"]["household"])
    school = float(config["transmission_probabilities"]["school"])
    classv = float(config["transmission_probabilities"]["class"])
    agec = float(config["transmission_probabilities"]["age_coupling_param"])
    mild = float(config["mild_detection_prob"])
    hosp = float(config["initial_conditions"]["hospitalization_multiplier"])
    precision = float(config["screening"]["precision"])
    trace_prob = float(config["household_params"]["trace_prob"])
    quarantine_prob = float(config["household_params"]["quarantine_prob"])

    out = Float64[]
    for day in 1:days
        bucket = min(cld(day, 30), length(infection))
        base = 15.0 * infection[bucket] * (0.6 + mild * detection[bucket])
        contact = 50.0 * (0.8 * household + 0.4 * school + 0.7 * classv + 0.5 * agec)
        control = 10.0 * tracing[bucket] * (trace_prob + quarantine_prob + precision)
        trend = 0.12 * day * infection[bucket]
        signal = max(0.0, base + contact + trend - control + 6.0 * hosp)
        push!(out, signal)
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# External simulation hook (single-run) invoking manager/MocosSimLauncher
# ─────────────────────────────────────────────────────────────────────────────

function run_external_sim(cfg::OptimizerConfig, candidate::Dict{String,Any}, days::Int; workdir::String)
    cfg.external_sim === nothing && error("External simulation config not provided")
    simcfg = cfg.external_sim
    mkpath(workdir)
    # write candidate config
    config_path = joinpath(workdir, "config_candidate.json")
    save_json(config_path, candidate)
    daily_path = joinpath(workdir, "output_daily.jld2")
    summary_path = joinpath(workdir, "summary.jld2")

    cmd = `$(simcfg.julia_bin) --project=$(simcfg.project_dir) --threads=4 $(simcfg.advanced_cli) $(config_path) --output-daily $(daily_path) --output-summary $(summary_path)`
    success = false
    try
        run(cmd)
        success = true
    catch err
        @warn "External simulation failed" err
    end
    return success, daily_path
end

function load_gt_series(gt_dir::String)
    function load_csv(name)
        path = joinpath(gt_dir, name)
        open(path, "r") do io
            first = true
            vals = Float64[]
            for line in eachline(io)
                first && (first = false; continue)
                parts = split(line, ",")
                length(parts) >= 2 || continue
                try
                    push!(vals, parse(Float64, parts[2]))
                catch
                end
            end
            return vals
        end
    end
    return Dict(
        "daily_detections" => load_csv("daily_detections.csv"),
        "daily_hospitalizations" => load_csv("daily_hospitalizations.csv"),
        "daily_deaths" => load_csv("daily_deaths.csv"),
    )
end

function read_daily_metric(path::String, metric::String)
    vals = Float64[]
    try
        h5open(path, "r") do h5
            for key in keys(h5)
                grp = h5[key]
                if haskey(grp, metric)
                    data = read(grp[metric])
                    append!(vals, Float64.(data))
                end
            end
        end
    catch err
        @warn "Failed to read HDF5 metric" path metric err
        return nothing
    end
    return vals
end

function rmse_series(a::Vector{Float64}, b::Vector{Float64})
    n = min(length(a), length(b))
    n == 0 && return Inf
    return sqrt(sum((a[i] - b[i])^2 for i in 1:n) / n)
end

function mae_series(a::Vector{Float64}, b::Vector{Float64})
    n = min(length(a), length(b))
    n == 0 && return Inf
    return sum(abs.(a[1:n] .- b[1:n])) / n
end

function rmae_series(a::Vector{Float64}, b::Vector{Float64})
    n = min(length(a), length(b))
    n == 0 && return Inf
    denom = max(mean(abs.(b[1:n])), 1e-9)
    return mae_series(a, b) / denom
end

function rolling_sum(series::Vector{Float64}, window::Int)
    length(series) == 0 && return Float64[]
    w = max(window, 1)
    out = similar(series)
    acc = 0.0
    for i in 1:length(series)
        acc += series[i]
        if i > w
            acc -= series[i - w]
        end
        out[i] = acc
    end
    return out
end

function score_with_real_sim(cfg::OptimizerConfig, candidate::Dict{String,Any}, days::Int; workdir::String)
    sim_ok, daily_path = run_external_sim(cfg, candidate, days; workdir=workdir)
    sim_ok || return Inf, Dict("sim_failed" => true)
    gt = load_gt_series(cfg.external_sim.gt_dir)
    metrics = Dict{String,Float64}()
    for (metric, gtvals) in gt
        simvals = read_daily_metric(daily_path, metric)
        if simvals === nothing
            metrics[metric] = Inf
            continue
        end
        g = Float64.(gtvals[1:min(end, days)])
        s = Float64.(simvals[1:min(end, days)])
        g = rolling_sum(g, 7)
        s = rolling_sum(s, 7)
        metrics[metric] = rmae_series(s, g)
    end
    combined = 0.5 * metrics["daily_detections"] + 0.0 * metrics["daily_hospitalizations"] + 1.0 * metrics["daily_deaths"]
    return combined, metrics
end

function score_from_daily(cfg::OptimizerConfig, daily_path::String, days::Int)
    gt = load_gt_series(cfg.external_sim.gt_dir)
    metrics = Dict{String,Float64}()
    det = read_daily_metric(daily_path, "daily_detections")
    hosp = read_daily_metric(daily_path, "daily_hospitalizations")
    deaths = read_daily_metric(daily_path, "daily_deaths")
    function rm(v, gkey)
        v === nothing && return Inf
        g = Float64.(gt[gkey][1:min(end, days)])
        s = Float64.(v[1:min(end, days)])
        g = rolling_sum(g, 7)
        s = rolling_sum(s, 7)
        rmae_series(s, g)
    end
    metrics["daily_detections"] = rm(det, "daily_detections")
    metrics["daily_hospitalizations"] = rm(hosp, "daily_hospitalizations")
    metrics["daily_deaths"] = rm(deaths, "daily_deaths")
    combined = 0.5 * metrics["daily_detections"] + 0.0 * metrics["daily_hospitalizations"] + 1.0 * metrics["daily_deaths"]
    return combined, metrics
end

function slurm_array_is_running(jobid::String)
    try
        out = read(`squeue -h -j $jobid -o "%.18i %.2t %.10M %.R"`, String)
        return !isempty(strip(out))
    catch
        return false
    end
end

function wait_for_iteration_outputs(list_file::String; poll::Float64=10.0)
    cand_dirs = [strip(x) for x in readlines(list_file) if !isempty(strip(x))]
    last_done = -1
    while true
        done_count = 0
        failed_count = 0
        pending = String[]
        for d in cand_dirs
            done_ok = isfile(joinpath(d, "done.ok"))
            failed_ok = isfile(joinpath(d, "failed.ok"))
            if done_ok
                done_count += 1
            elseif failed_ok
                failed_count += 1
            else
                push!(pending, d)
            end
        end

        if done_count + failed_count == length(cand_dirs)
            return
        end

        if done_count > 0 && done_count == last_done
            # no new completions for 1 minute -> mark remaining as failed and continue
            for d in pending
                failed_ok = joinpath(d, "failed.ok")
                isfile(failed_ok) || touch(failed_ok)
            end
            return
        end

        last_done = done_count
        sleep(60.0)
    end
end

function submit_slurm_array(cfg::OptimizerConfig, list_file::String)
    simcfg = cfg.external_sim
    lines = readlines(list_file)
    n = length(lines)
    n == 0 && return ""
    cmd = `sbatch --parsable -c 4 -t 01:00:00 --array=0-$(n-1) scripts/score_candidates.sh $list_file $(simcfg.julia_bin) $(simcfg.project_dir) $(simcfg.advanced_cli) $(simcfg.gt_dir)`
    out = strip(read(cmd, String))
    return out
end

function build_reference(seed::Dict{String,Any}, days::Int)
    return synthetic_simulation(seed, days)
end

function score_candidate(candidate::Dict{String,Any}, reference::Vector{Float64}, cfg::OptimizerConfig, days::Int; workdir::String="")
    if cfg.external_sim === nothing
        simulated = synthetic_simulation(candidate, days)
        full_daily = rmse(simulated, reference)
        recent_daily = recent_weighted_rmse(simulated, reference, cfg.objective.recent_days)
        full_daily > cfg.objective.early_reject_multiplier * max(rmse(reference, zeros(length(reference))), 1.0) && return Dict(
            "score" => 1.0e9,
            "full_daily" => full_daily,
            "recent_daily" => recent_daily,
            "early_reject" => true,
        )

        cumulative = abs(sum(simulated) - sum(reference)) / max(sum(reference), 1e-9)
        peak_height = abs(maximum(simulated) - maximum(reference)) / max(maximum(reference), 1e-9)
        peak_timing = abs(argmax(simulated) - argmax(reference)) / max(length(reference), 1)
        slope = rmse(finite_diff_slope(simulated), finite_diff_slope(reference))

        score = cfg.objective.weights["full_daily"] * full_daily +
                cfg.objective.weights["recent_daily"] * recent_daily +
                cfg.objective.weights["cumulative"] * cumulative +
                cfg.objective.weights["peak_height"] * peak_height +
                cfg.objective.weights["peak_timing"] * peak_timing +
                cfg.objective.weights["slope"] * slope

        return Dict(
            "score" => score,
            "full_daily" => full_daily,
            "recent_daily" => recent_daily,
            "cumulative" => cumulative,
            "peak_height" => peak_height,
            "peak_timing" => peak_timing,
            "slope" => slope,
            "early_reject" => false,
            "simulated" => "synthetic",
        )
    else
        workdir == "" && (workdir = mktempdir(prefix="simrun_"; parent=joinpath(cfg.output_dir, "real_sims")))
        combined, metrics = score_with_real_sim(cfg, candidate, days; workdir=workdir)
        return Dict(
            "score" => combined,
            "metrics" => metrics,
            "early_reject" => false,
            "simulated" => "real",
            "workdir" => workdir,
        )
    end
end

function cma_candidates(rng::AbstractRNG, state::CMAState, λ::Int)
    dim = length(state.mean)
    L = cholesky(Symmetric(state.covariance + 1e-6I)).L
    candidates = Vector{Vector{Float64}}(undef, λ)
    zs = Vector{Vector{Float64}}(undef, λ)
    for i in 1:λ
        z = randn(rng, dim)
        x = state.mean + state.sigma .* (L * z)
        candidates[i] = x
        zs[i] = z
    end
    return candidates, zs
end

function update_state(state::CMAState, ranked::Vector{Tuple{Float64,Vector{Float64},Vector{Float64}}})
    μ = max(2, length(ranked) ÷ 2)
    weights = [log(μ + 0.5) - log(i) for i in 1:μ]
    weights ./= sum(weights)
    selected = ranked[1:μ]
    new_mean = zeros(length(state.mean))
    for (w, (_, x, _)) in zip(weights, selected)
        new_mean .+= w .* x
    end
    centered = [x .- new_mean for (_, x, _) in selected]
    cov = zeros(size(state.covariance))
    for (w, c) in zip(weights, centered)
        cov .+= w .* (c * c')
    end
    new_cov = 0.7 .* state.covariance .+ 0.3 .* cov
    best_score = selected[1][1]
    new_sigma = best_score < ranked[min(end, μ)][1] ? state.sigma * 0.97 : state.sigma * 1.01
    return CMAState(new_mean, clamp(new_sigma, 0.02, 0.5), new_cov)
end

function run_stage(rng::AbstractRNG, seed::Dict{String,Any}, specs::Vector{ParamSpec}, cfg::OptimizerConfig, stage::StageConfig, state::Union{Nothing,CMAState}; use_slurm::Bool=false)
    active_months = stage.fit_months
    days = active_months * cfg.monthly_days
    reference = build_reference(seed, days)
    specs_stage = stage_specs(specs, cfg, stage)
    dim = sum(spec.length for spec in specs_stage)
    if state === nothing
        x0 = initial_vector(seed, specs_stage)
        state = CMAState(copy(x0), stage.sigma, Matrix{Float64}(I, dim, dim))
    else
        state = CMAState(copy(state.mean), stage.sigma, copy(state.covariance))
    end

    history = Any[]
    best_score = Inf
    best_vector = copy(state.mean)
    best_candidate = deepcopy(seed)

    for iter in 1:stage.max_iterations
        candidates, zs = cma_candidates(rng, state, stage.population_size)
        ranked = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
        # If external sim configured and slurm enabled, dispatch via Slurm array; otherwise score inline
        if cfg.external_sim !== nothing && use_slurm
            iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
            mkpath(iter_root)
            list_file = joinpath(iter_root, "candidate_list.txt")
            open(list_file, "w") do io
                for (ci, cand) in enumerate(candidates)
                    x = clip!(copy(cand), specs_stage)
                    candidate_cfg = vector_to_config(seed, specs_stage, x, active_months)
                    inject_frozen!(candidate_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                    cand_dir = joinpath(iter_root, @sprintf("cand_%02d", ci))
                    mkpath(cand_dir)
                    save_json(joinpath(cand_dir, "config.json"), candidate_cfg)
                    println(io, cand_dir)
                end
            end
            jobid = submit_slurm_array(cfg, list_file)
            @info "Submitted Slurm array" stage=stage.name iteration=iter jobid=jobid
            wait_for_iteration_outputs(list_file; poll=10.0)
            @info "Slurm array finished" stage=stage.name iteration=iter jobid=jobid

            # collect scores from generated output_daily.jld2
            for (ci, cand) in enumerate(candidates)
                x = clip!(copy(cand), specs_stage)
                cand_dir = joinpath(iter_root, @sprintf("cand_%02d", ci))
                daily_path = joinpath(cand_dir, "output_daily.jld2")
                cand_cfg = vector_to_config(seed, specs_stage, x, active_months)
                inject_frozen!(cand_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                metrics = if isfile(daily_path)
                    combined, comp = score_from_daily(cfg, daily_path, days)
                    metrics_payload = Dict(
                        "score" => combined,
                        "daily_detections" => comp["daily_detections"],
                        "daily_hospitalizations" => comp["daily_hospitalizations"],
                        "daily_deaths" => comp["daily_deaths"],
                        "simulated" => "real",
                    )
                    save_json(joinpath(cand_dir, "metrics.json"), metrics_payload)
                    Dict("score" => combined, "metrics" => comp, "simulated" => "real")
                else
                    metrics_payload = Dict("score" => Inf, "simulated" => "real_missing")
                    save_json(joinpath(cand_dir, "metrics.json"), metrics_payload)
                    Dict("score" => Inf, "metrics" => Dict(), "simulated" => "real_missing")
                end
                score = metrics["score"]
                push!(history, Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "fit_months" => active_months,
                    "score" => score,
                    "metrics" => metrics,
                ))
                push!(ranked, (score, x, zs[ci]))
                if score < best_score
                    best_score = score
                    best_vector = copy(x)
                    best_candidate = deepcopy(cand_cfg)
                end
            end
        else
            for (ci, cand) in enumerate(candidates)
                x = clip!(copy(cand), specs_stage)
                candidate_cfg = vector_to_config(seed, specs_stage, x, active_months)
                inject_frozen!(candidate_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                metrics = score_candidate(candidate_cfg, reference, cfg, days; workdir=joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)_cand_$(ci)"))
                save_json(joinpath(joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)_cand_$(ci)"), "metrics.json"), metrics)
                score = metrics["score"]
                push!(history, Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "fit_months" => active_months,
                    "score" => score,
                    "metrics" => metrics,
                ))
                push!(ranked, (score, x, zs[ci]))
                if score < best_score
                    best_score = score
                    best_vector = copy(x)
                    best_candidate = deepcopy(candidate_cfg)
                end
            end
        end
        sort!(ranked, by=first)
        state = update_state(state, ranked)
        state.mean .= best_vector
    end

    return Dict(
        "stage" => stage.name,
        "fit_months" => active_months,
        "best_score" => best_score,
        "best_candidate" => best_candidate,
        "best_vector" => best_vector,
        "sigma" => state.sigma,
        "covariance" => state.covariance,
        "history" => history,
    ), state
end

function run_optimizer(config_path::String; use_slurm::Bool=false)
    cfg = load_config(config_path)
    seed = load_json(cfg.seed_config)
    specs = build_specs(seed, cfg)
    rng = MersenneTwister(42)

    stage_outputs = Any[]
    all_history = Any[]
    state = nothing
    current_seed = deepcopy(seed)

    for stage in cfg.stages
        result, state = run_stage(rng, current_seed, specs, cfg, stage, state; use_slurm=use_slurm)
        current_seed = result["best_candidate"]
        update_stage_freeze!(cfg, stage, result["history"], specs)
        push!(stage_outputs, Dict(
            "stage" => result["stage"],
            "fit_months" => result["fit_months"],
            "best_score" => result["best_score"],
            "sigma" => result["sigma"],
        ))
        append!(all_history, result["history"])
        save_json(joinpath(cfg.output_dir, "$(stage.name)_best_candidate.json"), result["best_candidate"])
    end

    save_json(joinpath(cfg.output_dir, "optimizer_history.json"), all_history)
    save_json(joinpath(cfg.output_dir, "stage_summary.json"), stage_outputs)
    save_json(joinpath(cfg.output_dir, "final_best_candidate.json"), current_seed)
    return Dict("stage_summary" => stage_outputs, "output_dir" => cfg.output_dir)
end

function main()
    use_slurm = "--slurm" in ARGS
    config_args = filter(arg -> arg != "--slurm", ARGS)
    config_path = length(config_args) >= 1 ? config_args[1] : joinpath(dirname(@__DIR__), "optimizer_config.json")
    result = run_optimizer(config_path; use_slurm=use_slurm)
    println(JSON.json(result))
end

end
