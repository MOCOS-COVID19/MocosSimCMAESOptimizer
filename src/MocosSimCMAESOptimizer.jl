module MocosSimCMAESOptimizer

using JSON
using LinearAlgebra
using Random
using Statistics
using Dates
using HDF5
using Printf

const MANAGER_ROOT = abspath(joinpath(@__DIR__, ".."))
const CURRENT_OPTIMIZER_CONFIG = Ref{Any}(nothing)

export main, run_optimizer, run_long_horizon, run_nuts_from_archive,
       run_nuts_from_stage, posterior_reusable_state, safe_save_json

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
    top_k::Int
    min_completion_fraction::Float64
    finish_iter_delay::Int
    search_policy::String
end

struct PosteriorConfig
    enabled::Bool
    draws::Int
    warmup::Int
    max_depth::Int
    step_size::Float64
    temperature::Float64
end

struct OptimizerConfig
    seed_config::String
    output_dir::String
    monthly_days::Int
    stages::Vector{StageConfig}
    scalar_bounds::Dict{String,Tuple{Float64,Float64}}
    temporal_bounds::Dict{String,Tuple{Float64,Float64}}
    scalar_preprocessing::Dict{String,Dict{String,Any}}
    objective::ObjectiveConfig
    external_sim::Union{Nothing,ExternalSimConfig}
    stage_freeze::Dict{String,Vector{String}}
    initial_state::Union{Nothing,Dict{String,Any}}
    posterior::PosteriorConfig
end

include("posterior_sampler.jl")

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
    p_c::Vector{Float64}
    p_sigma::Vector{Float64}
end

CMAState(mean::Vector{Float64}, sigma::Float64, covariance::Matrix{Float64}) =
    CMAState(mean, sigma, covariance, zeros(length(mean)), zeros(length(mean)))

struct SearchPolicy
    name::String
    sigma_multiplier::Float64
    temporal_unlock_multiplier::Float64
    random_candidate_fraction::Float64
end

function get_search_policy(name::String)
    if name == "wide"
        return SearchPolicy("wide", 1.35, 1.4, 0.10)
    elseif name == "narrow"
        return SearchPolicy("narrow", 0.85, 0.9, 0.0)
    elseif name == "temporal_escape"
        return SearchPolicy("temporal_escape", 1.0, 1.8, 0.15)
    else
        return SearchPolicy("baseline", 1.0, 1.0, 0.0)
    end
end

function candidate_search_policy_names()
    return ["baseline", "wide", "narrow", "temporal_escape"]
end

function determine_search_policy(cfg::OptimizerConfig, stage::StageConfig)
    requested = cfg.objective.search_policy
    requested != "determine" && return get_search_policy(requested)
    # Lightweight automatic policy selection heuristic.
    # Prefer stronger exploration for later / harder horizons.
    if stage.fit_months >= 10
        return get_search_policy("temporal_escape")
    elseif stage.fit_months >= 8
        return get_search_policy("wide")
    else
        return get_search_policy("baseline")
    end
end

function full_reusable_state_from_cma(stage::StageConfig, specs_stage::Vector{ParamSpec}, state::CMAState)
    return Dict(
        "stage" => stage.name,
        "fit_months" => stage.fit_months,
        "param_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
        "param_ranges" => [spec.kind == :temporal ? [spec.lower, spec.upper] : [spec.lower, spec.upper] for spec in specs_stage],
        "mean" => state.mean,
        "sigma" => state.sigma,
        "covariance" => state.covariance,
        "p_c" => state.p_c,
        "p_sigma" => state.p_sigma,
    )
end

function stage_transition_state(prev::CMAState, stage::StageConfig, specs_stage::Vector{ParamSpec}; sigma_floor::Float64=0.08, sigma_scale::Float64=2.0)
    dim = length(prev.mean)
    cov = copy(prev.covariance)
    if size(cov, 1) != dim || size(cov, 2) != dim
        cov = Matrix{Float64}(I, dim, dim)
    end
    # Preserve the posterior-informed local geometry while adding a small
    # regularization term for the next stage.
    cov = 0.9 .* cov .+ 0.1 .* Matrix{Float64}(I, dim, dim)
    return CMAState(copy(prev.mean), max(stage.sigma, max(prev.sigma * sigma_scale, sigma_floor)), cov)
end

function initial_state_from_config(cfg::OptimizerConfig, dim::Int, default_mean::Vector{Float64})
    cfg.initial_state === nothing && return nothing
    raw = cfg.initial_state
    mean = haskey(raw, "mean") ? Float64.(raw["mean"]) : copy(default_mean)
    sigma = haskey(raw, "sigma") ? Float64(raw["sigma"]) : 0.3
    cov = if haskey(raw, "covariance")
        Matrix{Float64}(raw["covariance"])
    else
        Matrix{Float64}(I, dim, dim)
    end
    size(cov, 1) == dim && size(cov, 2) == dim || return nothing
    length(mean) == dim || return nothing
    p_c = haskey(raw, "p_c") ? Float64.(raw["p_c"]) : zeros(dim)
    p_sigma = haskey(raw, "p_sigma") ? Float64.(raw["p_sigma"]) : zeros(dim)
    length(p_c) == dim && length(p_sigma) == dim || return nothing
    return CMAState(mean, sigma, cov, p_c, p_sigma)
end

function load_full_reusable_state(path::String)
    isfile(path) || return nothing
    raw = load_json(path)
    haskey(raw, "param_names") && haskey(raw, "mean") && haskey(raw, "covariance") || return nothing
    return raw
end

function build_state_from_reusable(seed::Dict{String,Any}, specs_stage::Vector{ParamSpec}, reusable::Dict{String,Any}; sigma_floor::Float64=0.08, temporal_unlock_multiplier::Float64=1.0)
    old_names = [String(x) for x in reusable["param_names"]]
    old_mean = Float64.(reusable["mean"])
    old_cov = Matrix{Float64}(reusable["covariance"])
    old_sigma = haskey(reusable, "sigma") ? Float64(reusable["sigma"]) : 0.3

    new_names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    new_mean = initial_vector(seed, specs_stage)
    dim = length(new_names)
    new_cov = Matrix{Float64}(I, dim, dim)
    old_p_c = haskey(reusable, "p_c") ? Float64.(reusable["p_c"]) : zeros(length(old_names))
    old_p_sigma = haskey(reusable, "p_sigma") ? Float64.(reusable["p_sigma"]) : zeros(length(old_names))
    new_p_c = zeros(dim)
    new_p_sigma = zeros(dim)
    idx_map = Dict(name => i for (i, name) in enumerate(old_names))
    kept = Int[]
    for (j, name) in enumerate(new_names)
        haskey(idx_map, name) || continue
        i = idx_map[name]
        new_mean[j] = old_mean[i]
        i <= length(old_p_c) && (new_p_c[j] = old_p_c[i])
        i <= length(old_p_sigma) && (new_p_sigma[j] = old_p_sigma[i])
        push!(kept, i)
    end
    for (a, name_a) in enumerate(new_names), (b, name_b) in enumerate(new_names)
        haskey(idx_map, name_a) && haskey(idx_map, name_b) || continue
        ia = idx_map[name_a]
        ib = idx_map[name_b]
        new_cov[a, b] = old_cov[ia, ib]
    end
    # Phase 1 temporal unlock on resume:
    # keep scalar warm-starts stable, but loosen temporal params so the search can
    # escape minima inherited from shorter horizons / older resumed states.
    temporal_mean_jitter = 0.12 * temporal_unlock_multiplier
    temporal_covariance_inflation = 2.5 * temporal_unlock_multiplier
    idx = 1
    for spec in specs_stage
        if spec.kind == :temporal
            for local_idx in 1:spec.length
                pos = idx + local_idx - 1
                # Later temporal buckets get stronger unlock.
                frac = spec.length <= 1 ? 1.0 : (local_idx - 1) / (spec.length - 1)
                jitter_scale = temporal_mean_jitter * (0.5 + frac)
                new_mean[pos] = clamp(
                    new_mean[pos] + jitter_scale * (2rand() - 1),
                    spec.lower,
                    spec.upper,
                )
                new_cov[pos, pos] = max(new_cov[pos, pos] * temporal_covariance_inflation * (1.0 + frac), 1e-6)
            end
        end
        idx += spec.length
    end
    return CMAState(new_mean, max(old_sigma, sigma_floor), new_cov, new_p_c, new_p_sigma)
end

function temporal_unlock_from_top_candidates!(state::CMAState, specs_stage::Vector{ParamSpec}, top_candidates)
    top_candidates === nothing && return state
    top_candidates isa AbstractVector || return state
    isempty(top_candidates) && return state

    # Phase 2:
    # Use spread among top candidates as a mismatch / uncertainty proxy.
    # Buckets that vary a lot across strong candidates get unlocked more aggressively.
    idx = 1
    for spec in specs_stage
        if spec.kind == :temporal
            bucket_values = [Float64[] for _ in 1:spec.length]
            for cand in top_candidates
                cfg = get(cand, "config", nothing)
                cfg isa AbstractDict || continue
                vals = try
                    get_nested(cfg, spec.name)
                catch
                    nothing
                end
                vals isa AbstractVector || continue
                for bi in 1:min(spec.length, length(vals))
                    v = vals[bi]
                    v isa Number || continue
                    push!(bucket_values[bi], Float64(v))
                end
            end
            for bi in 1:spec.length
                pos = idx + bi - 1
                vals = bucket_values[bi]
                isempty(vals) && continue
                spread = length(vals) == 1 ? 0.0 : std(vals)
                frac = spec.length <= 1 ? 1.0 : (bi - 1) / (spec.length - 1)
                unlock = clamp(spread / max(spec.upper - spec.lower, 1e-6), 0.0, 1.0)
                jitter_scale = 0.05 + 0.20 * unlock + 0.08 * frac
                state.mean[pos] = clamp(
                    state.mean[pos] + jitter_scale * (2rand() - 1),
                    spec.lower,
                    spec.upper,
                )
                inflate = 1.0 + 2.0 * unlock + 0.5 * frac
                state.covariance[pos, pos] = max(state.covariance[pos, pos] * inflate, 1e-6)
            end
        end
        idx += spec.length
    end
    return state
end

function temporal_unlock_from_bucket_errors!(state::CMAState, specs_stage::Vector{ParamSpec}, top_candidates)
    top_candidates === nothing && return state
    top_candidates isa AbstractVector || return state
    isempty(top_candidates) && return state

    best = top_candidates[1]
    metrics = get(best, "metrics", nothing)
    metrics isa AbstractDict || return state
    bucket_errors = get(metrics, "bucket_errors", nothing)
    bucket_errors isa AbstractDict || return state

    idx = 1
    for spec in specs_stage
        if spec.kind == :temporal && haskey(bucket_errors, spec.name)
            errors = try
                Float64.(bucket_errors[spec.name])
            catch
                Float64[]
            end
            if !isempty(errors)
                max_err = max(maximum(errors), 1e-9)
                for bi in 1:min(spec.length, length(errors))
                    pos = idx + bi - 1
                    frac = spec.length <= 1 ? 1.0 : (bi - 1) / (spec.length - 1)
                    unlock = clamp(errors[bi] / max_err, 0.0, 1.0)
                    jitter_scale = 0.04 + 0.22 * unlock + 0.06 * frac
                    state.mean[pos] = clamp(
                        state.mean[pos] + jitter_scale * (2rand() - 1),
                        spec.lower,
                        spec.upper,
                    )
                    inflate = 1.0 + 2.5 * unlock + 0.5 * frac
                    state.covariance[pos, pos] = max(state.covariance[pos, pos] * inflate, 1e-6)
                end
            end
        end
        idx += spec.length
    end
    return state
end

function parse_stage_iter_from_path(path::String)
    m = match(r"stage_(\d+).*/iter_(\d+)", replace(path, '\\' => '/'))
    m === nothing && return 0, 0
    return parse(Int, m.captures[1]), parse(Int, m.captures[2])
end

function append_jsonl(path::String, value)
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, JSON.json(value))
    end
end

function load_stage_state(stage_root::String)
    path = joinpath(stage_root, "stage_state.json")
    isfile(path) || return nothing
    return load_json(path)
end

function latest_iteration_top_candidates(stage_root::String)
    isdir(stage_root) || return nothing
    iter_dirs = String[]
    for entry in readdir(stage_root)
        startswith(entry, "iter_") || continue
        path = joinpath(stage_root, entry, "top_candidates.json")
        isfile(path) && push!(iter_dirs, path)
    end
    isempty(iter_dirs) && return nothing
    function iter_num(path::String)
        m = match(r"iter_(\d+)", path)
        m === nothing && return 0
        return parse(Int, m.captures[1])
    end
    best_path = iter_dirs[1]
    best_iter = iter_num(best_path)
    for path in iter_dirs[2:end]
        cur = iter_num(path)
        if cur > best_iter
            best_iter = cur
            best_path = path
        end
    end
    return load_json(best_path)
end

function stage_resume_info(stage_root::String)
    iter_infos = Vector{Tuple{Int,String,Bool}}()
    isdir(stage_root) || return nothing
    for entry in readdir(stage_root)
        startswith(entry, "iter_") || continue
        iter_dir = joinpath(stage_root, entry)
        cand = joinpath(iter_dir, "candidate_list.txt")
        m = match(r"iter_(\d+)", entry)
        m === nothing && continue
        iter_idx = parse(Int, m.captures[1])
        top_candidates_file = joinpath(iter_dir, "top_candidates.json")
        has_any_candidate_dirs = any(startswith(name, "cand_") for name in readdir(iter_dir))
        completed = isfile(top_candidates_file) || has_any_candidate_dirs
        isfile(cand) && push!(iter_infos, (iter_idx, cand, completed))
    end
    isempty(iter_infos) && return nothing
    completed_iters = [info for info in iter_infos if info[3]]
    function max_iter_info(items)
        best = items[1]
        for item in items[2:end]
            if item[1] > best[1]
                best = item
            end
        end
        return best
    end
    chosen = isempty(completed_iters) ? max_iter_info(iter_infos) : max_iter_info(completed_iters)
    return Dict(
        "last_iter" => chosen[1],
        "last_iter_file" => chosen[2],
        "iteration_completed" => chosen[3],
        "found_iterations" => sort([info[1] for info in iter_infos]),
        "completed_iterations" => sort([info[1] for info in completed_iters]),
    )
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
    config_dir = dirname(abspath(path))
    stages = [StageConfig(s["name"], s["fit_months"], s["max_iterations"], s["population_size"], float(s["sigma"])) for s in raw["stages"]]
    scalar_bounds = Dict(k => (float(v[1]), float(v[2])) for (k, v) in raw["scalar_bounds"])
    temporal_bounds = Dict(k => (float(v[1]), float(v[2])) for (k, v) in raw["temporal_bounds"])
    scalar_preprocessing = Dict{String,Dict{String,Any}}()
    if haskey(raw, "scalar_preprocessing")
        for (k, v) in raw["scalar_preprocessing"]
            scalar_preprocessing[String(k)] = Dict(String(kk) => vv for (kk, vv) in v)
        end
    end
    objective = ObjectiveConfig(
        Dict(k => float(v) for (k, v) in raw["objective"]["weights"]),
        Int(get(raw["objective"], "top_k", 1)),
        float(get(raw["objective"], "min_completion_fraction", 1.0)),
        Int(get(raw["objective"], "finish_iter_delay", 30)),
        String(get(raw["objective"], "search_policy", "baseline")),
    )
    posterior_raw = get(raw, "posterior", Dict{String,Any}())
    posterior = PosteriorConfig(
        Bool(get(posterior_raw, "enabled", true)),
        Int(get(posterior_raw, "draws", 500)),
        Int(get(posterior_raw, "warmup", 250)),
        Int(get(posterior_raw, "max_depth", 8)),
        float(get(posterior_raw, "step_size", 0.05)),
        float(get(posterior_raw, "temperature", 1.0)),
    )
    gt_dir = haskey(raw, "gt_dir") ?
        (isabspath(String(raw["gt_dir"])) ?
            String(raw["gt_dir"]) :
            normpath(joinpath(config_dir, String(raw["gt_dir"])))) :
        nothing
    external_sim = gt_dir !== nothing && haskey(raw, "julia_bin") && haskey(raw, "project_dir") && haskey(raw, "advanced_cli") ?
        ExternalSimConfig(gt_dir, String(raw["julia_bin"]), String(raw["project_dir"]), String(raw["advanced_cli"])) :
        nothing
    stage_freeze = Dict{String,Vector{String}}()
    if haskey(raw, "stage_freeze")
        for (k, v) in raw["stage_freeze"]
            stage_freeze[String(k)] = [String(x) for x in v]
        end
    end
    initial_state = haskey(raw, "initial_state") ? Dict{String,Any}(String(k) => v for (k, v) in raw["initial_state"]) : nothing
    return OptimizerConfig(raw["seed_config"], raw["output_dir"], Int(raw["monthly_days"]), stages, scalar_bounds, temporal_bounds, scalar_preprocessing, objective, external_sim, stage_freeze, initial_state, posterior)
end

function scalar_preprocessing_entry(cfg::OptimizerConfig, spec::ParamSpec)
    return get(cfg.scalar_preprocessing, spec.name, nothing)
end

function encode_scalar_value(cfg::OptimizerConfig, seed::Dict{String,Any}, spec::ParamSpec, value)
    preprocessing = scalar_preprocessing_entry(cfg, spec)
    preprocessing === nothing && return float(value)
    mode = get(preprocessing, "mode", nothing)
    mode == "normalize_to_bounds" || return float(value)
    lo = float(get(preprocessing, "min", spec.lower))
    hi = float(get(preprocessing, "max", spec.upper))
    hi > lo || error("normalize_to_bounds requires max > min for $(spec.name)")
    return clamp((float(value) - lo) / (hi - lo), 0.0, 1.0)
end

function decode_scalar_value(cfg::OptimizerConfig, seed::Dict{String,Any}, spec::ParamSpec, value::Float64)
    preprocessing = scalar_preprocessing_entry(cfg, spec)
    raw = value
    if preprocessing !== nothing
        mode = get(preprocessing, "mode", nothing)
        if mode == "normalize_to_bounds"
            lo = float(get(preprocessing, "min", spec.lower))
            hi = float(get(preprocessing, "max", spec.upper))
            hi > lo || error("normalize_to_bounds requires max > min for $(spec.name)")
            raw = lo + clamp(value, 0.0, 1.0) * (hi - lo)
        end
    end
    map_mode = preprocessing === nothing ? nothing : get(preprocessing, "map", nothing)
    if map_mode == "integer" || endswith(spec.name, ".num_infections") || occursin(".time_limit", spec.name)
        return round(Int, raw)
    end
    return raw
end

function get_nested(config::Dict{String,Any}, path::String)
    node = config
    parts = split(path, ".")
    for (i, part) in enumerate(parts)
        m = match(r"^([^\[]+)\[(\d+)\]$", part)
        if m !== nothing
            key = m.captures[1]
            idx = parse(Int, m.captures[2])
            idx > 0 || error("Config path uses zero-based index in '$path'. Use Julia-style 1-based indexing.")
            node = node[key]
            if i == length(parts)
                return node[idx]
            end
            node = node[idx]
            continue
        end
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
        m = match(r"^([^\[]+)\[(\d+)\]$", part)
        if m !== nothing
            key = m.captures[1]
            idx = parse(Int, m.captures[2])
            idx > 0 || error("Config path uses zero-based index in '$path'. Use Julia-style 1-based indexing.")
            node = node[key][idx]
        else
            node = node[part]
        end
    end
    last = parts[end]
    m = match(r"^([^\[]+)\[(\d+)\]$", last)
    if m !== nothing
        key = m.captures[1]
        idx = parse(Int, m.captures[2])
        idx > 0 || error("Config path uses zero-based index in '$path'. Use Julia-style 1-based indexing.")
        node[key][idx] = value
    else
        node[last] = value
    end
    return config
end

function build_specs(seed::Dict{String,Any}, cfg::OptimizerConfig)
    specs = ParamSpec[]
    for (name, (lo, hi)) in sort(collect(cfg.scalar_bounds), by=first)
        preprocessing = get(cfg.scalar_preprocessing, name, nothing)
        if preprocessing !== nothing && get(preprocessing, "mode", nothing) == "normalize_to_bounds"
            push!(specs, ParamSpec(name, :scalar, 1, 0.0, 1.0))
        else
            push!(specs, ParamSpec(name, :scalar, 1, lo, hi))
        end
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
    optcfg = CURRENT_OPTIMIZER_CONFIG[]
    for spec in specs
        current = get_nested(seed, spec.name)
        if spec.kind == :scalar
            val = optcfg === nothing ? float(current) : encode_scalar_value(optcfg, seed, spec, current)
            push!(values, val)
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

function temporal_active_length(seed::Dict{String,Any}, spec::ParamSpec, active_months::Int, cfg::OptimizerConfig)
    active_days = active_months * cfg.monthly_days
    interval_times = get_nested(seed, replace(spec.name, "interval_values" => "interval_times"))
    isempty(interval_times) && return min(active_days, spec.length)
    if length(interval_times) == 1
        step_days = max(float(interval_times[1]), 1.0)
    else
        deltas = [float(interval_times[i+1]) - float(interval_times[i]) for i in 1:length(interval_times)-1]
        positive_deltas = [d for d in deltas if d > 0]
        step_days = isempty(positive_deltas) ? max(float(interval_times[1]), 1.0) : minimum(positive_deltas)
    end
    return min(cld(active_days, max(round(Int, step_days), 1)), spec.length)
end

function temporal_bucket_day_ranges(seed::Dict{String,Any}, spec::ParamSpec, active_months::Int, cfg::OptimizerConfig)
    active_days = active_months * cfg.monthly_days
    interval_times = get_nested(seed, replace(spec.name, "interval_values" => "interval_times"))
    isempty(interval_times) && return [(1, active_days) for _ in 1:spec.length]
    ranges = Tuple{Int,Int}[]
    for i in 1:spec.length
        start_day = i == 1 ? 1 : max(1, round(Int, float(interval_times[min(i, length(interval_times))])))
        end_day = i == spec.length ? active_days : max(start_day, round(Int, float(interval_times[min(i, length(interval_times))])))
        push!(ranges, (start_day, min(end_day, active_days)))
    end
    return ranges
end

function vector_to_config(seed::Dict{String,Any}, specs::Vector{ParamSpec}, x::Vector{Float64}, active_months::Int)
    cfg = deepcopy(seed)
    optcfg = CURRENT_OPTIMIZER_CONFIG[]
    idx = 1
    for spec in specs
        if spec.kind == :scalar
            val = optcfg === nothing ? x[idx] : decode_scalar_value(optcfg, seed, spec, x[idx])
            set_nested!(cfg, spec.name, val)
            idx += 1
        else
            current = map(float, get_nested(cfg, spec.name))
            active = optcfg === nothing ? min(active_months, spec.length) : temporal_active_length(seed, spec, active_months, optcfg)
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
        rows = Vector{Tuple{Int,Union{Missing,Float64}}}()
        open(path, "r") do io
            first = true
            for line in eachline(io)
                first && (first = false; continue)
                parts = split(line, ",")
                length(parts) >= 2 || continue
                day = try
                    parse(Int, strip(parts[1]))
                catch
                    continue
                end
                value = try
                    parse(Float64, strip(parts[2]))
                catch
                    missing
                end
                push!(rows, (day, value))
            end
        end
        isempty(rows) && return Float64[]
        sort!(rows, by = first)
        min_day = rows[1][1]
        max_day = rows[end][1]
        values = Union{Missing,Float64}[missing for _ in min_day:max_day]
        for (day, value) in rows
            values[day - min_day + 1] = value
        end
        return values
    end
    function load_optional(name)
        path = joinpath(gt_dir, name)
        isfile(path) || return Float64[]
        return load_csv(name)
    end
    return Dict(
        "daily_detections" => load_csv("daily_age_total_detections.csv"),
        "daily_hospitalizations" => load_csv("daily_hospitalizations.csv"),
        "daily_deaths" => load_csv("daily_age_total_deaths.csv"),
        "daily_student_detections" => load_csv("sax-scholars-infections-normalized.csv"),
        "daily_age_total_detections" => load_optional("daily_age_total_detections.csv"),
        "daily_age_00_04_detections" => load_optional("daily_age_00_04_detections.csv"),
        "daily_age_05_14_detections" => load_optional("daily_age_05_14_detections.csv"),
        "daily_age_15_34_detections" => load_optional("daily_age_15_34_detections.csv"),
        "daily_age_35_59_detections" => load_optional("daily_age_35_59_detections.csv"),
        "daily_age_60_79_detections" => load_optional("daily_age_60_79_detections.csv"),
        "daily_age_80_plus_detections" => load_optional("daily_age_80_plus_detections.csv"),
        "daily_age_total_deaths" => load_optional("daily_age_total_deaths.csv"),
        "daily_age_00_04_deaths" => load_optional("daily_age_00_04_deaths.csv"),
        "daily_age_05_14_deaths" => load_optional("daily_age_05_14_deaths.csv"),
        "daily_age_15_34_deaths" => load_optional("daily_age_15_34_deaths.csv"),
        "daily_age_35_59_deaths" => load_optional("daily_age_35_59_deaths.csv"),
        "daily_age_60_79_deaths" => load_optional("daily_age_60_79_deaths.csv"),
        "daily_age_80_plus_deaths" => load_optional("daily_age_80_plus_deaths.csv"),
        "household_infections" => load_optional("household_infections.csv"),
        "household_infection_rate" => load_optional("household_infection_rate.csv"),
    )
end

function moving_average(series::Vector{Float64}, window::Int=7)
    n = length(series)
    n == 0 && return Float64[]
    w = max(window, 1)
    out = similar(series)
    acc = 0.0
    for i in 1:n
        acc += series[i]
        if i > w
            acc -= series[i - w]
        end
        out[i] = acc / min(i, w)
    end
    return out
end

function read_daily_metric(path::String, metric::String)
    series = Vector{Vector{Float64}}()
    isfile(path) || return nothing
    try
        h5open(path, "r") do h5
            for key in sort(collect(keys(h5)))
                grp = h5[key]
                if haskey(grp, metric)
                    data = read(grp[metric])
                    push!(series, Float64.(vec(data)))
                end
            end
        end
    catch err
        @warn "Failed to read HDF5 metric" path metric err
        return nothing
    end
    isempty(series) && return nothing
    return series
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

function cumulative_series(series::Vector{Float64})
    out = similar(series)
    acc = 0.0
    for i in 1:length(series)
        acc += series[i]
        out[i] = acc
    end
    return out
end

function drop_missing(series)
    return Float64[float(x) for x in series if x !== missing]
end

"""
Compute 7-day-window GT alignment value for student detections with missing-aware rules:
- If 7 values are missing (all 7 are missing) => ignore this window (returns `nothing`).
- If 6 values are missing => use 1/7 of the single non-missing value.
- Otherwise (<=5 missings) => use the mean over available values (equally-weighted).
"""
function student_window_gt_value(window::AbstractVector{T}; expected_window::Int = 7) where {T<:Union{Missing,Float64}}
    miss = 0
    sumv = 0.0
    cnt = 0
    for x in window
        if x === missing
            miss += 1
        else
            v = Float64(x)
            sumv += v
            cnt += 1
        end
    end
    if miss == expected_window
        return nothing
    elseif cnt == 0
        return nothing
    else
        # Missing-aware weekly value: sum of available GT values scaled by 1/7
        # (matches rule: sum(vi)/7 for all non-missing vi)
        return (sumv / expected_window)
    end
end

"""
Compute student detection weekly targets/sim values on 7-day windows with missing-aware GT weighting.
Returns vectors of equal length containing only windows that are not ignored.
"""
function student_weekly_aligned_vectors(gt_daily::AbstractVector{T}, sim_daily::AbstractVector{Float64}, days::Int) where {T<:Union{Missing,Float64}}
    n = min(days, length(sim_daily), length(gt_daily))
    if n <= 0
        return Float64[], Float64[]
    end
    out_g = Float64[]
    out_s = Float64[]
    # windows are [i-6..i] with i being 1-based index end of window
    w = 7
    for end_idx in 1:n
        start_idx = end_idx - w + 1
        start_idx = max(start_idx, 1)
        window_gt = gt_daily[start_idx:end_idx]
        # for sim we still take 7-day mean over the available prefix; this matches existing "rolling mean" behavior
        window_sim = sim_daily[start_idx:end_idx]
        # Only apply missing-aware rule when we have the full 7-day window; for partial start, follow same logic with shorter window
        expected = w
        val_g = student_window_gt_value(window_gt; expected_window=expected)
        if val_g === nothing
            continue
        end
        # sim weekly value: mean over available sim values in window
        val_s = mean(window_sim)
        push!(out_g, Float64(val_g))
        push!(out_s, val_s)
    end
    return out_g, out_s
end

function per_trajectory_rmae(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Inf
    vals = Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = metric == "daily_student_detections" ? rolling_mean(g, 7) : rolling_sum(g, 7)
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        if metric == "daily_student_detections"
            # Missing-aware weekly alignment:
            # build weekly vectors from the *original* gt_series (with missing), but sim already has numbers.
            gt_daily = gt_series[1:min(end, days)]
            sim_daily = traj[1:min(end, days)]
            gg, ss = student_weekly_aligned_vectors(gt_daily, Float64.(sim_daily), days)
            if isempty(gg)
                continue
            end
            push!(vals, rmae_series(ss, gg))
        else
            s = rolling_sum(s, 7)
            push!(vals, rmae_series(s, g))
        end
    end
    isempty(vals) && return Inf
    return sum(vals) / length(vals)
end

function per_trajectory_cumulative_error(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Inf
    vals = Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = cumulative_series(g)
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        s = cumulative_series(s)
        push!(vals, abs(last(s) - last(g)) / max(abs(last(g)), 1e-9))
    end
    isempty(vals) && return Inf
    return sum(vals) / length(vals)
end

function rolling_mean(series::Vector{Float64}, window::Int)
    length(series) == 0 && return Float64[]
    w = max(window, 1)
    out = similar(series)
    acc = 0.0
    for i in 1:length(series)
        acc += series[i]
        if i > w
            acc -= series[i - w]
        end
        out[i] = acc / min(i, w)
    end
    return out
end

function trajectory_metric_values(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = rolling_sum(g, 7)
    vals = Float64[]
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        s = rolling_sum(s, 7)
        push!(vals, rmae_series(s, g))
    end
    return vals
end

function weekly_error_distributions(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Dict{String,Any}(
        "absolute_error" => Float64[],
        "normalized_absolute_error" => Float64[],
        "mae" => Float64[],
        "rmae" => Float64[],
        "periods" => Vector{Vector{Int}}(),
        "observations" => Float64[],
        "predictions" => Float64[],
    )
    n = min(days, length(gt_series))
    absolute_errors = Float64[]
    normalized_absolute_errors = Float64[]
    maes = Float64[]
    rmaes = Float64[]
    periods = Vector{Vector{Int}}()
    observations = Float64[]
    predictions = Float64[]
    week_size = 7
    for start_day in 1:week_size:n
        end_day = min(start_day + week_size - 1, n)
        valid_days = [day for day in start_day:end_day if gt_series[day] !== missing]
        isempty(valid_days) && continue
        absolute_error = 0.0
        normalized_absolute_error = 0.0
        mae = 0.0
        rmae = 0.0
        counted = 0
        for traj in trajs
            length(traj) < first(valid_days) && continue
            sim_week = Float64[traj[day] for day in valid_days if day <= length(traj)]
            paired_gt = Float64[gt_series[day] for day in valid_days if day <= length(traj)]
            isempty(sim_week) && continue
            absolute_error += abs(sum(sim_week) - sum(paired_gt))
            normalized_absolute_error += abs(sum(sim_week) - sum(paired_gt)) /
                                        max(sum(abs.(paired_gt)), 1.0)
            trajectory_mae = mean(abs.(sim_week .- paired_gt))
            mae += trajectory_mae
            rmae += trajectory_mae / max(mean(abs.(paired_gt)), 1e-9)
            counted += 1
        end
        if counted > 0
            push!(absolute_errors, absolute_error / counted)
            push!(normalized_absolute_errors, normalized_absolute_error / counted)
            push!(maes, mae / counted)
            push!(rmaes, rmae / counted)
            push!(periods, [start_day, end_day])
            push!(observations, sum(gt_series[valid_days]))
            push!(predictions, mean([
                sum(Float64[traj[day] for day in valid_days if day <= length(traj)])
                for traj in trajs if any(day <= length(traj) for day in valid_days)
            ]))
        end
    end
    return Dict{String,Any}(
        "absolute_error" => absolute_errors,
        "normalized_absolute_error" => normalized_absolute_errors,
        "mae" => maes,
        "rmae" => rmaes,
        "periods" => periods,
        "observations" => observations,
        "predictions" => predictions,
    )
end

const WEEKLY_CONTROL_METRICS = [
    "daily_age_total_detections",
    "daily_age_total_deaths",
    "daily_age_00_04_detections",
    "daily_age_05_14_detections",
    "daily_age_15_34_detections",
    "daily_age_35_59_detections",
    "daily_age_60_79_detections",
    "daily_age_80_plus_detections",
    "daily_age_00_04_deaths",
    "daily_age_05_14_deaths",
    "daily_age_15_34_deaths",
    "daily_age_35_59_deaths",
    "daily_age_60_79_deaths",
    "daily_age_80_plus_deaths",
]

function weekly_control_score(daily_path::String, gt::AbstractDict, days::Int)
    metric_scores = Float64[]
    for metric in WEEKLY_CONTROL_METRICS
        haskey(gt, metric) || continue
        errors = weekly_error_distributions(daily_path, metric, gt[metric], days)
        rmaes = errors["rmae"]
        normalized_absolute_errors = errors["normalized_absolute_error"]
        isempty(rmaes) && continue
        n = min(length(rmaes), length(normalized_absolute_errors))
        score = mean(0.7 .* rmaes[1:n] .+ 0.3 .* normalized_absolute_errors[1:n])
        isfinite(score) && push!(metric_scores, score)
    end
    isempty(metric_scores) && return Inf
    return mean(metric_scores)
end

function weekly_control_bucket_errors(
    daily_path::String,
    gt::AbstractDict,
    days::Int,
    spec::ParamSpec,
    active_months::Int,
    cfg::OptimizerConfig,
)
    bucket_ranges = temporal_bucket_day_ranges(load_json(cfg.seed_config), spec, active_months, cfg)
    errors_by_metric = Dict{String,Any}()
    for metric in WEEKLY_CONTROL_METRICS
        haskey(gt, metric) || continue
        errors_by_metric[metric] = weekly_error_distributions(daily_path, metric, gt[metric], days)
    end
    errors = zeros(Float64, spec.length)
    for (bi, (start_day, end_day)) in enumerate(bucket_ranges)
        bucket_values = Float64[]
        for metric_errors in values(errors_by_metric)
            periods = metric_errors["periods"]
            rmaes = metric_errors["rmae"]
            normalized_absolute_errors = metric_errors["normalized_absolute_error"]
            for (pi, period) in enumerate(periods)
                period[1] <= end_day && period[2] >= start_day || continue
                pi > length(rmaes) && continue
                pi > length(normalized_absolute_errors) && continue
                push!(bucket_values, 0.7 * rmaes[pi] + 0.3 * normalized_absolute_errors[pi])
            end
        end
        errors[bi] = isempty(bucket_values) ? 0.0 : mean(bucket_values)
    end
    return errors
end

function cumulative_metric_values(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = cumulative_series(g)
    vals = Float64[]
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        s = cumulative_series(s)
        push!(vals, abs(last(s) - last(g)) / max(abs(last(g)), 1e-9))
    end
    return vals
end

function cumulative_error_distribution(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = cumulative_series(g)
    denom = max(abs(last(g)), 1e-9)
    vals = Float64[]
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        s = cumulative_series(s)
        push!(vals, abs(last(s) - last(g)) / denom)
    end
    return vals
end

function household_readout(daily_path::String, days::Int)
    trajs = read_daily_metric(daily_path, "daily_detections")
    trajs === nothing && return Dict{String,Float64}("household_infections" => NaN, "household_infection_rate" => NaN)
    # Approximation: use detection trajectories as a proxy and provide a normalized readout.
    # The LASUB household statistics are known to be biased, so treat this as graphable
    # simulation telemetry rather than a ground-truth calibrated estimate.
    household_proxy = Float64[]
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        push!(household_proxy, sum(s) / max(length(s), 1))
    end
    infections = mean(household_proxy)
    rate = infections / max(days, 1)
    return Dict("household_infections" => infections, "household_infection_rate" => rate)
end

function temporal_directional_guidance(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int, spec::ParamSpec, active_months::Int, cfg::OptimizerConfig)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = rolling_sum(g, 7)
    bucket_ranges = temporal_bucket_day_ranges(load_json(cfg.seed_config), spec, active_months, cfg)
    guidance = zeros(Float64, spec.length)
    for (bi, (start_day, end_day)) in enumerate(bucket_ranges)
        start_day > end_day && continue
        idxs = start_day:min(end_day, length(g))
        isempty(idxs) && continue
        g_seg = g[idxs]
        err = 0.0
        for traj in trajs
            s = Float64.(traj[1:min(end, days)])
            s = rolling_sum(s, 7)
            s_seg = s[idxs]
            err += mean(abs.(s_seg .- g_seg))
        end
        guidance[bi] = err / length(trajs)
    end
    return guidance
end

function temporal_bucket_error_distribution(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int, spec::ParamSpec, active_months::Int, cfg::OptimizerConfig)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    if metric == "daily_student_detections"
        g = rolling_mean(g, 7)
    else
        g = rolling_sum(g, 7)
    end
    bucket_ranges = temporal_bucket_day_ranges(load_json(cfg.seed_config), spec, active_months, cfg)
    errors = zeros(Float64, spec.length)
    for (bi, (start_day, end_day)) in enumerate(bucket_ranges)
        start_day > end_day && continue
        idxs = start_day:min(end_day, length(g))
        isempty(idxs) && continue
        g_seg = g[idxs]
        err = 0.0
        counted = 0
        for traj in trajs
            s = Float64.(traj[1:min(end, days)])
            if metric == "daily_student_detections"
                s = rolling_mean(s, 7)
            else
                s = rolling_sum(s, 7)
            end
            s_seg = s[idxs]
            n = min(length(s_seg), length(g_seg))
            n == 0 && continue
            err += mean(abs.(s_seg[1:n] .- g_seg[1:n]))
            counted += 1
        end
        errors[bi] = counted == 0 ? 0.0 : err / counted
    end
    return errors
end

function score_with_real_sim(cfg::OptimizerConfig, candidate::Dict{String,Any}, days::Int; workdir::String)
    sim_ok, daily_path = run_external_sim(cfg, candidate, days; workdir=workdir)
    sim_ok || return Inf, Dict("sim_failed" => true)
    gt = load_gt_series(cfg.external_sim.gt_dir)
    weekly_control = weekly_control_score(daily_path, gt, days)
    metrics = Dict{String,Float64}()
    for (metric, gtvals) in gt
        isempty(gtvals) && continue
        metrics[metric] = per_trajectory_rmae(daily_path, metric, drop_missing(gtvals), days)
        metrics["$(metric)_cumulative"] = per_trajectory_cumulative_error(daily_path, metric, drop_missing(gtvals), days)
    end
    weights = cfg.objective.weights
    combined = get(weights, "daily_detections", 1.0) * get(metrics, "daily_detections", 0.0) +
               get(weights, "daily_hospitalizations", 0.0) * get(metrics, "daily_hospitalizations", 0.0) +
               get(weights, "daily_deaths", 1.0) * get(metrics, "daily_deaths", 0.0) +
               get(weights, "daily_student_detections", 1.0) * get(metrics, "daily_student_detections", 0.0) +
               get(weights, "daily_detections_cumulative", 1.0) * get(metrics, "daily_detections_cumulative", 0.0) +
               get(weights, "daily_hospitalizations_cumulative", 0.0) * get(metrics, "daily_hospitalizations_cumulative", 0.0) +
               get(weights, "daily_deaths_cumulative", 1.0) * get(metrics, "daily_deaths_cumulative", 0.0) +
               get(weights, "daily_student_detections_cumulative", 1.0) * get(metrics, "daily_student_detections_cumulative", 0.0) +
               get(weights, "daily_age_00_04_detections", 0.0) * get(metrics, "daily_age_00_04_detections", 0.0) +
               get(weights, "daily_age_05_14_detections", 0.0) * get(metrics, "daily_age_05_14_detections", 0.0) +
               get(weights, "daily_age_15_34_detections", 0.0) * get(metrics, "daily_age_15_34_detections", 0.0) +
               get(weights, "daily_age_35_59_detections", 0.0) * get(metrics, "daily_age_35_59_detections", 0.0) +
               get(weights, "daily_age_60_79_detections", 0.0) * get(metrics, "daily_age_60_79_detections", 0.0) +
               get(weights, "daily_age_80_plus_detections", 0.0) * get(metrics, "daily_age_80_plus_detections", 0.0) +
               get(weights, "daily_age_00_04_deaths", 0.0) * get(metrics, "daily_age_00_04_deaths", 0.0) +
               get(weights, "daily_age_05_14_deaths", 0.0) * get(metrics, "daily_age_05_14_deaths", 0.0) +
               get(weights, "daily_age_15_34_deaths", 0.0) * get(metrics, "daily_age_15_34_deaths", 0.0) +
               get(weights, "daily_age_35_59_deaths", 0.0) * get(metrics, "daily_age_35_59_deaths", 0.0) +
               get(weights, "daily_age_60_79_deaths", 0.0) * get(metrics, "daily_age_60_79_deaths", 0.0) +
               get(weights, "daily_age_80_plus_deaths", 0.0) * get(metrics, "daily_age_80_plus_deaths", 0.0) +
               get(weights, "weekly_control", 1.0) * weekly_control
    metrics["weekly_control_score"] = weekly_control
    return combined, metrics
end

function score_from_daily(cfg::OptimizerConfig, daily_path::String, days::Int)
    gt = load_gt_series(cfg.external_sim.gt_dir)
    weekly_control = weekly_control_score(daily_path, gt, days)
    metrics = Dict{String,Float64}()
    for (metric, gtvals) in gt
        isempty(gtvals) && continue
        metrics[metric] = per_trajectory_rmae(daily_path, metric, gtvals, days)
        metrics["$(metric)_cumulative"] = per_trajectory_cumulative_error(daily_path, metric, gtvals, days)
    end
    weights = cfg.objective.weights
    combined = get(weights, "daily_detections", 1.0) * get(metrics, "daily_detections", 0.0) +
               get(weights, "daily_hospitalizations", 0.0) * get(metrics, "daily_hospitalizations", 0.0) +
               get(weights, "daily_deaths", 1.0) * get(metrics, "daily_deaths", 0.0) +
               get(weights, "daily_student_detections", 1.0) * get(metrics, "daily_student_detections", 0.0) +
               get(weights, "daily_detections_cumulative", 1.0) * get(metrics, "daily_detections_cumulative", 0.0) +
               get(weights, "daily_hospitalizations_cumulative", 0.0) * get(metrics, "daily_hospitalizations_cumulative", 0.0) +
               get(weights, "daily_deaths_cumulative", 1.0) * get(metrics, "daily_deaths_cumulative", 0.0) +
               get(weights, "daily_student_detections_cumulative", 0.0) * get(metrics, "daily_student_detections_cumulative", 0.0) +
               get(weights, "daily_age_00_04_detections", 0.0) * get(metrics, "daily_age_00_04_detections", 0.0) +
               get(weights, "daily_age_05_14_detections", 0.0) * get(metrics, "daily_age_05_14_detections", 0.0) +
               get(weights, "daily_age_15_34_detections", 0.0) * get(metrics, "daily_age_15_34_detections", 0.0) +
               get(weights, "daily_age_35_59_detections", 0.0) * get(metrics, "daily_age_35_59_detections", 0.0) +
               get(weights, "daily_age_60_79_detections", 0.0) * get(metrics, "daily_age_60_79_detections", 0.0) +
               get(weights, "daily_age_80_plus_detections", 0.0) * get(metrics, "daily_age_80_plus_detections", 0.0) +
               get(weights, "daily_age_00_04_deaths", 0.0) * get(metrics, "daily_age_00_04_deaths", 0.0) +
               get(weights, "daily_age_05_14_deaths", 0.0) * get(metrics, "daily_age_05_14_deaths", 0.0) +
               get(weights, "daily_age_15_34_deaths", 0.0) * get(metrics, "daily_age_15_34_deaths", 0.0) +
               get(weights, "daily_age_35_59_deaths", 0.0) * get(metrics, "daily_age_35_59_deaths", 0.0) +
               get(weights, "daily_age_60_79_deaths", 0.0) * get(metrics, "daily_age_60_79_deaths", 0.0) +
               get(weights, "daily_age_80_plus_deaths", 0.0) * get(metrics, "daily_age_80_plus_deaths", 0.0) +
               get(weights, "weekly_control", 1.0) * weekly_control
    metrics["weekly_control_score"] = weekly_control
    return combined, metrics
end

function top_k_entries(entries, k::Int)
    items = collect(entries)
    isempty(items) && return Any[]
    kk = max(1, min(k, length(items)))
    sorted = sort(items, by = x -> Float64(get(x, "score", Inf)))
    return sorted[1:kk]
end

function with_iteration_ranks(entries)
    ranked = sort(collect(entries), by = x -> Float64(get(x, "score", Inf)))
    out = Any[]
    for (idx, entry) in enumerate(ranked)
        enriched = deepcopy(entry)
        enriched["rank_within_iteration"] = idx
        push!(out, enriched)
    end
    return out
end

function safe_iteration_top_k(entries, k::Int)
    try
        return top_k_entries(with_iteration_ranks(entries), k)
    catch err
        @error "Failed to build iteration top candidates" err entries_count=length(entries)
        fallback = Any[]
        ranked = sort(collect(entries), by = x -> Float64(get(x, "score", Inf)))
        kk = max(1, min(k, length(ranked)))
        for (idx, entry) in enumerate(ranked[1:kk])
            enriched = deepcopy(entry)
            enriched["rank_within_iteration"] = idx
            push!(fallback, enriched)
        end
        return fallback
    end
end

function slurm_array_is_running(jobid::String)
    try
        out = read(`squeue -h -j $jobid -o "%.18i %.2t %.10M %.R"`, String)
        return !isempty(strip(out))
    catch
        return false
    end
end

function wait_for_iteration_outputs(list_file::String; poll::Float64=10.0, min_completion_fraction::Float64=1.0, finish_iter_delay::Int=30)
    cand_dirs = [strip(x) for x in readlines(list_file) if !isempty(strip(x))]
    target_done = max(1, ceil(Int, length(cand_dirs) * clamp(min_completion_fraction, 0.0, 1.0)))
    threshold_reached_at = nothing
    while true
        done_count = 0
        failed_count = 0
        pending = String[]
        for d in cand_dirs
            done_ok = isfile(joinpath(d, "done.ok"))
            failed_ok = isfile(joinpath(d, "failed.ok"))
            skipped_ok = isfile(joinpath(d, "skipped.ok"))
            if done_ok
                done_count += 1
            elseif failed_ok
                failed_count += 1
            elseif skipped_ok
                failed_count += 1
            else
                push!(pending, d)
            end
        end

        if done_count + failed_count == length(cand_dirs)
            return Dict(
                "done" => done_count,
                "failed" => failed_count,
                "pending" => pending,
                "pending_count" => length(pending),
                "threshold_reached" => done_count >= target_done,
                "iteration_truncated" => false,
            )
        end

        if done_count >= target_done
            if threshold_reached_at === nothing
                threshold_reached_at = time()
            end
            if done_count + failed_count == length(cand_dirs)
                return Dict(
                    "done" => done_count,
                    "failed" => failed_count,
                    "pending" => pending,
                    "pending_count" => length(pending),
                    "threshold_reached" => true,
                    "iteration_truncated" => false,
                )
            end
            if (time() - threshold_reached_at) >= finish_iter_delay
                for d in pending
                    skipped_ok = joinpath(d, "skipped.ok")
                    isfile(skipped_ok) || touch(skipped_ok)
                end
                return Dict(
                    "done" => done_count,
                    "failed" => failed_count,
                    "pending" => pending,
                    "pending_count" => length(pending),
                    "threshold_reached" => true,
                    "iteration_truncated" => !isempty(pending),
                )
            end
        end

        if threshold_reached_at !== nothing && finish_iter_delay <= 0
            for d in pending
                skipped_ok = joinpath(d, "skipped.ok")
                isfile(skipped_ok) || touch(skipped_ok)
            end
            return Dict(
                "done" => done_count,
                "failed" => failed_count,
                "pending" => pending,
                "pending_count" => length(pending),
                "threshold_reached" => true,
                "iteration_truncated" => !isempty(pending),
            )
        end

        sleep(poll)
    end
end

function submit_slurm_array(cfg::OptimizerConfig, list_file::String)
    simcfg = cfg.external_sim
    lines = readlines(list_file)
    n = length(lines)
    n == 0 && return ""
    cmd = `sbatch --parsable -c 4 -t 01:00:00 --mem=20G --array=0-$(n-1) scripts/score_candidates.sh $list_file $(simcfg.julia_bin) $(simcfg.project_dir) $(simcfg.advanced_cli) $(simcfg.gt_dir)`
    last_err = nothing
    for attempt in 1:5
        try
            out = strip(read(cmd, String))
            isempty(out) && error("empty sbatch output")
            return out
        catch err
            last_err = err
            @warn "sbatch submission failed, retrying" attempt err
            sleep(5.0 * attempt)
        end
    end
    error("Failed to submit Slurm array after retries: $(last_err)")
end

function cancel_slurm_array(jobid)
    jobid_str = String(jobid)
    isempty(strip(jobid_str)) && return
    try
        run(`scancel $jobid_str`)
    catch err
        @warn "Failed to cancel Slurm array job" jobid=jobid_str err
    end
end

function score_candidate(candidate::Dict{String,Any}, cfg::OptimizerConfig, days::Int; workdir::String="")
    workdir == "" && (workdir = mktempdir(prefix="simrun_"; parent=joinpath(cfg.output_dir, "real_sims")))
    combined, metrics = score_with_real_sim(cfg, candidate, days; workdir=workdir)
    result = Dict(
        "score" => combined,
        "metrics" => metrics,
        "early_reject" => false,
        "simulated" => "real",
        "workdir" => workdir,
    )
    if cfg.external_sim !== nothing
        daily_path = joinpath(workdir, "output_daily.jld2")
        if isfile(daily_path)
            weekly_absolute_errors = Dict{String,Any}()
            weekly_normalized_absolute_errors = Dict{String,Any}()
            weekly_mae = Dict{String,Any}()
            weekly_rmae = Dict{String,Any}()
            weekly_error_periods = Dict{String,Any}()
            weekly_observations = Dict{String,Any}()
            weekly_predictions = Dict{String,Any}()
            for (metric, gtvals) in load_gt_series(cfg.external_sim.gt_dir)
                errors = weekly_error_distributions(daily_path, metric, gtvals, days)
                weekly_absolute_errors[metric] = errors["absolute_error"]
                weekly_normalized_absolute_errors[metric] = errors["normalized_absolute_error"]
                weekly_mae[metric] = errors["mae"]
                weekly_rmae[metric] = errors["rmae"]
                weekly_error_periods[metric] = errors["periods"]
                weekly_observations[metric] = errors["observations"]
                weekly_predictions[metric] = errors["predictions"]
            end
            result["weekly_absolute_errors"] = weekly_absolute_errors
            result["weekly_normalized_absolute_errors"] = weekly_normalized_absolute_errors
            result["weekly_mae"] = weekly_mae
            result["weekly_rmae"] = weekly_rmae
            result["weekly_error_periods"] = weekly_error_periods
            result["weekly_observations"] = weekly_observations
            result["weekly_predictions"] = weekly_predictions
        end
    end
    return result
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

function clip_candidate(candidate::Vector{Float64}, specs_stage::Vector{ParamSpec})
    evaluated = copy(candidate)
    lower_hits = Bool[]
    upper_hits = Bool[]
    idx = 1
    for spec in specs_stage
        for _ in 1:spec.length
            value = evaluated[idx]
            push!(lower_hits, value < spec.lower)
            push!(upper_hits, value > spec.upper)
            evaluated[idx] = clamp(value, spec.lower, spec.upper)
            idx += 1
        end
    end
    return evaluated, Dict(
        "clipped" => any(lower_hits) || any(upper_hits),
        "lower_bound_hits" => lower_hits,
        "upper_bound_hits" => upper_hits,
    )
end

function cma_diagnostics(state::CMAState)
    values = eigen(Symmetric(state.covariance)).values
    values = max.(Float64.(values), 1e-12)
    return Dict(
        "eigenvalues" => values,
        "covariance_trace" => tr(state.covariance),
        "covariance_condition_number" => maximum(values) / minimum(values),
        "p_c_norm" => norm(state.p_c),
        "p_sigma_norm" => norm(state.p_sigma),
    )
end

function update_state(state::CMAState, ranked::Vector{Tuple{Float64,Vector{Float64},Vector{Float64}}})
    n = length(state.mean)
    λ = length(ranked)
    μ = max(2, λ ÷ 2)
    weights = [log(μ + 0.5) - log(i) for i in 1:μ]
    weights ./= sum(weights)
    μ_eff = 1.0 / sum(weights .^ 2)
    c_sigma = (μ_eff + 2.0) / (n + μ_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, sqrt((μ_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + μ_eff / n) / (n + 4.0 + 2.0 * μ_eff / n)
    c_1 = 2.0 / ((n + sqrt(2.0))^2 + μ_eff)
    c_mu = min(1.0 - c_1, 2.0 * (μ_eff - 2.0 + 1.0 / μ_eff) / ((n + 2.0)^2 + μ_eff))
    selected = ranked[1:μ]
    new_mean = zeros(length(state.mean))
    for (w, (_, x, _)) in zip(weights, selected)
        new_mean .+= w .* x
    end
    y_w = (new_mean .- state.mean) ./ max(state.sigma, 1e-12)
    eig = eigen(Symmetric(state.covariance + 1e-10I))
    eigvals = max.(eig.values, 1e-10)
    invsqrt_c = eig.vectors * Diagonal(1.0 ./ sqrt.(eigvals)) * eig.vectors'
    p_sigma = (1.0 - c_sigma) .* state.p_sigma .+
              sqrt(c_sigma * (2.0 - c_sigma) * μ_eff) .* (invsqrt_c * y_w)
    chi_n = sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n^2))
    norm_p_sigma = norm(p_sigma)
    h_sigma = norm_p_sigma / sqrt(1.0 - (1.0 - c_sigma)^(2.0)) <
              (1.4 + 2.0 / (n + 1.0)) * chi_n
    p_c = (1.0 - c_c) .* state.p_c .+
          (h_sigma ? 1.0 : 0.0) * sqrt(c_c * (2.0 - c_c) * μ_eff) .* y_w
    rank_mu_cov = zeros(size(state.covariance))
    for (w, (_, x, _)) in zip(weights, selected)
        y = (x .- state.mean) ./ max(state.sigma, 1e-12)
        rank_mu_cov .+= w .* (y * y')
    end
    correction = (1.0 - h_sigma) * c_c * (2.0 - c_c)
    new_cov = (1.0 - c_1 - c_mu) .* state.covariance .+
              c_1 .* (p_c * p_c' + correction .* state.covariance) .+
              c_mu .* rank_mu_cov
    new_cov = 0.5 .* (new_cov + new_cov')
    new_sigma = state.sigma * exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1.0))
    return CMAState(new_mean, clamp(new_sigma, 0.02, 0.5), new_cov, p_c, p_sigma)
end

function safe_save_json(path::String, value; label::String=path)
    try
        save_json(path, value)
    catch err
        @error "Failed to save JSON artifact" label path err
        rethrow(err)
    end
end

function append_cma_candidate_record(
    iter_root::String,
    stage::StageConfig,
    iteration::Int,
    candidate_id::Int,
    raw_candidate::Vector{Float64},
    evaluated_candidate::Vector{Float64},
    z::Vector{Float64},
    clip_info::Dict{String,Any},
    state::CMAState,
    score,
    metrics_path::String,
)
    append_jsonl(joinpath(iter_root, "posterior_training_data.jsonl"), Dict(
        "stage" => stage.name,
        "iteration" => iteration,
        "candidate" => candidate_id,
        "x_raw" => raw_candidate,
        "x_evaluated" => evaluated_candidate,
        "z" => z,
        "clipping" => clip_info,
        "score" => score,
        "metrics_path" => metrics_path,
        "simulation_distribution" => Dict(
            "mean" => state.mean,
            "sigma" => state.sigma,
            "covariance" => state.covariance,
        ),
    ))
end


function long_horizon_stage_name(days::Int, monthly_days::Int)
    months = ceil(Int, days / max(monthly_days, 1))
    return "stable_$(months)m"
end

function run_long_horizon(cfg_path::String; days::Int=730, output_dir::Union{Nothing,String}=nothing, seed_config::Union{Nothing,String}=nothing)
    cfg = load_config(cfg_path)
    root = output_dir === nothing ? joinpath(cfg.output_dir, "stable_long_run") : output_dir
    mkpath(root)
    seed_path = seed_config === nothing ? cfg.seed_config : seed_config
    seed = load_json(seed_path)
    stage_name = long_horizon_stage_name(days, cfg.monthly_days)
    stage = StageConfig(stage_name, ceil(Int, days / max(cfg.monthly_days, 1)), 1, 1, 0.0)
    specset = load_param_specs(seed, cfg, stage)
    state = nothing
    rng = MersenneTwister(42)
    result, _ = run_stage(rng, seed, specset, cfg, stage, state; use_slurm=false, resume_from=0)
    safe_save_json(joinpath(root, "long_horizon_summary.json"), Dict(
        "days" => days,
        "stage" => stage_name,
        "result" => result,
    ); label="long_horizon_summary")
    return result
end
function run_stage(rng::AbstractRNG, seed::Dict{String,Any}, specs::Vector{ParamSpec}, cfg::OptimizerConfig, stage::StageConfig, state::Union{Nothing,CMAState}; use_slurm::Bool=false, resume_from::Int=0)
    active_months = stage.fit_months
    days = active_months * cfg.monthly_days
    policy = determine_search_policy(cfg, stage)
    specs_stage = stage_specs(specs, cfg, stage)
    dim = sum(spec.length for spec in specs_stage)
    if state === nothing
        x0 = initial_vector(seed, specs_stage)
        reusable_path = joinpath(cfg.output_dir, "full_reusable_state.json")
        reusable = load_full_reusable_state(reusable_path)
        if reusable !== nothing
            state = build_state_from_reusable(seed, specs_stage, reusable; temporal_unlock_multiplier=policy.temporal_unlock_multiplier)
            stage_root = joinpath(cfg.output_dir, "real_sims", stage.name)
            top_candidates = latest_iteration_top_candidates(stage_root)
            state = temporal_unlock_from_bucket_errors!(state, specs_stage, top_candidates)
        else
            seeded = initial_state_from_config(cfg, dim, x0)
            state = seeded === nothing ? CMAState(copy(x0), stage.sigma, Matrix{Float64}(I, dim, dim)) : seeded
        end
    else
        state = stage_transition_state(state, stage, specs_stage; sigma_scale=1.25, sigma_floor=0.02)
    end
    state = CMAState(
        copy(state.mean),
        state.sigma * policy.sigma_multiplier,
        copy(state.covariance),
        copy(state.p_c),
        copy(state.p_sigma),
    )

    stage_root = joinpath(cfg.output_dir, "real_sims", stage.name)
    resume_state = load_stage_state(stage_root)
    history = Any[]
    iter_log = Any[]
    top_candidates = Dict{String,Dict{String,Any}}()
    best_score_raw = resume_state === nothing ? Inf : get(resume_state, "best_score", Inf)
    best_score = best_score_raw === nothing ? Inf : Float64(best_score_raw)
    best_vector = resume_state !== nothing && haskey(resume_state, "best_vector") ? Float64.(resume_state["best_vector"]) : copy(state.mean)
    best_candidate = deepcopy(seed)
    if resume_state !== nothing && haskey(resume_state, "best_vector")
        sigma_raw = get(resume_state, "sigma", state.sigma)
        p_c = haskey(resume_state, "p_c") ? Float64.(resume_state["p_c"]) : zeros(dim)
        p_sigma = haskey(resume_state, "p_sigma") ? Float64.(resume_state["p_sigma"]) : zeros(dim)
        state = CMAState(copy(best_vector), sigma_raw === nothing ? state.sigma : Float64(sigma_raw), state.covariance, p_c, p_sigma)
    end
    start_iter = max(1, resume_from + 1)
    @info "Starting stage run" stage=stage.name fit_months=active_months start_iter=start_iter max_iterations=stage.max_iterations population_size=stage.population_size use_slurm=use_slurm search_policy=policy.name

    for iter in start_iter:stage.max_iterations
        @info "Starting iteration" stage=stage.name iteration=iter sigma=state.sigma best_score=best_score
        candidates, zs = cma_candidates(rng, state, stage.population_size)
        iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
        mkpath(iter_root)
        safe_save_json(joinpath(iter_root, "cma_sampling_state.json"), Dict(
            "stage" => stage.name,
            "iteration" => iter,
            "param_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
            "mean" => state.mean,
            "sigma" => state.sigma,
            "covariance" => state.covariance,
            "p_c" => state.p_c,
            "p_sigma" => state.p_sigma,
            "diagnostics" => cma_diagnostics(state),
        ); label="cma_sampling_state")
        ranked = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
        iteration_top_candidates = Any[]
        # If external sim configured and slurm enabled, dispatch via Slurm array; otherwise score inline
        if cfg.external_sim !== nothing && use_slurm
            iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
            mkpath(iter_root)
            list_file = joinpath(iter_root, "candidate_list.txt")
            open(list_file, "w") do io
                for (ci, cand) in enumerate(candidates)
                    x, clip_info = clip_candidate(cand, specs_stage)
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
            wait_result = wait_for_iteration_outputs(
                list_file;
                poll=10.0,
                min_completion_fraction=cfg.objective.min_completion_fraction,
                finish_iter_delay=cfg.objective.finish_iter_delay,
            )
            @info "Iteration wait result" stage=stage.name iteration=iter jobid=jobid completed_count=wait_result["done"] failed_count=wait_result["failed"] pending_count=wait_result["pending_count"] threshold_reached=wait_result["threshold_reached"] iteration_truncated=wait_result["iteration_truncated"]
            if get(wait_result, "iteration_truncated", false)
                cancel_slurm_array(jobid)
            end
            @info "Slurm array finished" stage=stage.name iteration=iter jobid=jobid

            # collect scores from generated output_daily.jld2
            for (ci, cand) in enumerate(candidates)
                x, clip_info = clip_candidate(cand, specs_stage)
                cand_dir = joinpath(iter_root, @sprintf("cand_%02d", ci))
                daily_path = joinpath(cand_dir, "output_daily.jld2")
                cand_cfg = vector_to_config(seed, specs_stage, x, active_months)
                inject_frozen!(cand_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                skipped = isfile(joinpath(cand_dir, "skipped.ok"))
                metrics = if skipped
                    metrics_payload = Dict("score" => Inf, "simulated" => "real_skipped", "status" => "skipped")
                    safe_save_json(joinpath(cand_dir, "metrics.json"), metrics_payload; label="candidate_metrics")
                    Dict("score" => Inf, "metrics" => Dict(), "simulated" => "real_skipped", "status" => "skipped")
                elseif isfile(daily_path)
                    combined, comp = score_from_daily(cfg, daily_path, days)
                    gt = load_gt_series(cfg.external_sim.gt_dir)
                    bucket_errors = Dict{String,Any}()
                    weekly_absolute_errors = Dict{String,Any}()
                    weekly_normalized_absolute_errors = Dict{String,Any}()
                    weekly_mae = Dict{String,Any}()
                    weekly_rmae = Dict{String,Any}()
                    weekly_error_periods = Dict{String,Any}()
                    weekly_observations = Dict{String,Any}()
                    weekly_predictions = Dict{String,Any}()
                    household = household_readout(daily_path, days)
                    for (metric, gtvals) in gt
                        weekly_errors = weekly_error_distributions(daily_path, metric, gtvals, days)
                        weekly_absolute_errors[metric] = weekly_errors["absolute_error"]
                        weekly_normalized_absolute_errors[metric] = weekly_errors["normalized_absolute_error"]
                        weekly_mae[metric] = weekly_errors["mae"]
                        weekly_rmae[metric] = weekly_errors["rmae"]
                        weekly_error_periods[metric] = weekly_errors["periods"]
                        weekly_observations[metric] = weekly_errors["observations"]
                        weekly_predictions[metric] = weekly_errors["predictions"]
                    end
                    metrics_payload = Dict(
                        "score" => combined,
                        "daily_detections" => comp["daily_detections"],
                        "daily_hospitalizations" => comp["daily_hospitalizations"],
                        "daily_deaths" => comp["daily_deaths"],
                        "daily_detections_cumulative" => comp["daily_detections_cumulative"],
                        "daily_hospitalizations_cumulative" => comp["daily_hospitalizations_cumulative"],
                        "daily_deaths_cumulative" => comp["daily_deaths_cumulative"],
                        "weekly_control_score" => comp["weekly_control_score"],
                        "daily_detections_per_trajectory" => trajectory_metric_values(daily_path, "daily_detections", gt["daily_detections"], days),
                        "daily_hospitalizations_per_trajectory" => trajectory_metric_values(daily_path, "daily_hospitalizations", gt["daily_hospitalizations"], days),
                        "daily_deaths_per_trajectory" => trajectory_metric_values(daily_path, "daily_deaths", gt["daily_deaths"], days),
                        "daily_detections_cumulative_per_trajectory" => cumulative_error_distribution(daily_path, "daily_detections", gt["daily_detections"], days),
                        "daily_hospitalizations_cumulative_per_trajectory" => cumulative_error_distribution(daily_path, "daily_hospitalizations", gt["daily_hospitalizations"], days),
                        "daily_deaths_cumulative_per_trajectory" => cumulative_error_distribution(daily_path, "daily_deaths", gt["daily_deaths"], days),
                        "weekly_absolute_errors" => weekly_absolute_errors,
                        "weekly_normalized_absolute_errors" => weekly_normalized_absolute_errors,
                        "weekly_mae" => weekly_mae,
                        "weekly_rmae" => weekly_rmae,
                        "weekly_error_periods" => weekly_error_periods,
                        "weekly_observations" => weekly_observations,
                        "weekly_predictions" => weekly_predictions,
                        "household_infections" => household["household_infections"],
                        "household_infection_rate" => household["household_infection_rate"],
                        "simulated" => "real",
                    )
                    for spec in specs_stage
                        spec.kind == :temporal || continue
                        bucket_errors[spec.name] = weekly_control_bucket_errors(
                            daily_path, gt, days, spec, active_months, cfg
                        )
                    end
                    metrics_payload["bucket_errors"] = bucket_errors
                    if haskey(comp, "daily_student_detections")
                        metrics_payload["daily_student_detections"] = comp["daily_student_detections"]
                        metrics_payload["daily_student_detections_cumulative"] = get(comp, "daily_student_detections_cumulative", NaN)
                        metrics_payload["daily_student_detections_per_trajectory"] = trajectory_metric_values(daily_path, "daily_student_detections", gt["daily_student_detections"], days)
                        metrics_payload["daily_student_detections_cumulative_per_trajectory"] = cumulative_error_distribution(daily_path, "daily_student_detections", gt["daily_student_detections"], days)
                    end
                    if haskey(comp, "household_infections")
                        metrics_payload["household_infections"] = comp["household_infections"]
                        metrics_payload["household_infection_rate"] = get(comp, "household_infection_rate", NaN)
                    end
                    safe_save_json(joinpath(cand_dir, "metrics.json"), metrics_payload; label="candidate_metrics")
                    Dict("score" => combined, "metrics" => comp, "simulated" => "real", "status" => "completed")
                else
                    metrics_payload = Dict("score" => Inf, "simulated" => "real_missing", "status" => "failed")
                    safe_save_json(joinpath(cand_dir, "metrics.json"), metrics_payload; label="candidate_metrics")
                    Dict("score" => Inf, "metrics" => Dict(), "simulated" => "real_missing", "status" => "failed")
                end
                score = metrics["score"]
                if get(metrics, "status", "completed") == "completed" && isfinite(Float64(score))
                    append_cma_candidate_record(
                        iter_root, stage, iter, ci, cand, x, zs[ci], clip_info,
                        state, score, joinpath(cand_dir, "metrics.json")
                    )
                end
                append_jsonl(joinpath(stage_root, "iter_metrics.jsonl"), Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "search_policy" => policy.name,
                    "score" => score,
                    "simulated" => metrics["simulated"],
                    "status" => get(metrics, "status", "unknown"),
                    "has_output_daily" => isfile(daily_path),
                    "sigma" => state.sigma,
                    "best_score_so_far" => best_score,
                    "threshold_reached" => wait_result["threshold_reached"],
                    "iteration_truncated" => wait_result["iteration_truncated"],
                    "completed_count" => wait_result["done"],
                    "failed_count" => wait_result["failed"],
                    "pending_count" => wait_result["pending_count"],
                ))
                push!(history, Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "search_policy" => policy.name,
                    "fit_months" => active_months,
                    "score" => score,
                    "status" => get(metrics, "status", "unknown"),
                    "metrics" => metrics,
                ))
                key = "$(iter)-$(ci)"
                candidate_entry = Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "search_policy" => policy.name,
                    "fit_months" => active_months,
                    "score" => score,
                    "status" => get(metrics, "status", "unknown"),
                    "config" => deepcopy(cand_cfg),
                    "metrics" => metrics,
                )
                top_candidates[key] = candidate_entry
                push!(iteration_top_candidates, candidate_entry)
                if get(metrics, "status", "failed") == "completed"
                    push!(ranked, (score, cand, zs[ci]))
                end
                if get(metrics, "status", "failed") == "completed" && score < best_score
                    best_score = score
                    best_vector = copy(x)
                    best_candidate = deepcopy(cand_cfg)
                end
            end
        else
            for (ci, cand) in enumerate(candidates)
                iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
                mkpath(iter_root)
                x, clip_info = clip_candidate(cand, specs_stage)
                candidate_cfg = vector_to_config(seed, specs_stage, x, active_months)
                inject_frozen!(candidate_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                metrics = score_candidate(candidate_cfg, cfg, days; workdir=joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)_cand_$(ci)"))
                safe_save_json(joinpath(joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)_cand_$(ci)"), "metrics.json"), metrics; label="candidate_metrics")
                score = metrics["score"]
                if get(metrics, "status", "completed") == "completed" && isfinite(Float64(score))
                    append_cma_candidate_record(
                        iter_root, stage, iter, ci, cand, x, zs[ci], clip_info,
                        state, score, joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)_cand_$(ci)", "metrics.json")
                    )
                end
                push!(history, Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "search_policy" => policy.name,
                    "fit_months" => active_months,
                    "score" => score,
                    "metrics" => metrics,
                ))
                key = "$(iter)-$(ci)"
                candidate_entry = Dict(
                    "stage" => stage.name,
                    "iteration" => iter,
                    "candidate" => ci,
                    "search_policy" => policy.name,
                    "fit_months" => active_months,
                    "score" => score,
                    "config" => deepcopy(candidate_cfg),
                    "metrics" => metrics,
                )
                top_candidates[key] = candidate_entry
                push!(iteration_top_candidates, candidate_entry)
                push!(ranked, (score, cand, zs[ci]))
                if score < best_score
                    best_score = score
                    best_vector = copy(x)
                    best_candidate = deepcopy(candidate_cfg)
                end
            end
        end
        isempty(ranked) && error("No completed candidates available for stage $(stage.name) iteration $(iter). Increase max wait or completion fraction.")
        sort!(ranked, by=first)
        @info "Updating CMA state" stage=stage.name iteration=iter completed_candidates=length(ranked) best_iteration_score=ranked[1][1]
        state = update_state(state, ranked)
        push!(iter_log, Dict(
            "stage" => stage.name,
            "iteration" => iter,
            "search_policy" => policy.name,
            "best_score" => best_score,
            "sigma" => state.sigma,
            "covariance_trace" => tr(state.covariance),
        ))
        safe_save_json(joinpath(stage_root, "stage_state.json"), Dict(
            "stage" => stage.name,
            "search_policy" => policy.name,
            "fit_months" => active_months,
            "best_score" => best_score,
            "sigma" => state.sigma,
            "covariance_trace" => tr(state.covariance),
            "covariance" => state.covariance,
            "p_c" => state.p_c,
            "p_sigma" => state.p_sigma,
            "cma_diagnostics" => cma_diagnostics(state),
            "best_vector" => best_vector,
        ); label="stage_state")
        iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
        mkpath(iter_root)
        safe_save_json(joinpath(iter_root, "top_candidates.json"), safe_iteration_top_k(iteration_top_candidates, cfg.objective.top_k); label="iteration_top_candidates")
        safe_save_json(joinpath(stage_root, "full_reusable_state.json"), full_reusable_state_from_cma(stage, specs_stage, state); label="full_reusable_state")
        @info "Finished iteration" stage=stage.name iteration=iter best_score=best_score sigma=state.sigma top_candidates_written=length(iteration_top_candidates)
    end

    @info "Finished stage run" stage=stage.name best_score=best_score iterations_run=(start_iter > stage.max_iterations ? 0 : stage.max_iterations - start_iter + 1)

    return Dict(
        "stage" => stage.name,
        "search_policy" => policy.name,
        "fit_months" => active_months,
        "best_score" => best_score,
        "best_candidate" => best_candidate,
        "top_candidates" => top_k_entries(collect(values(top_candidates)), cfg.objective.top_k),
        "best_vector" => best_vector,
        "sigma" => state.sigma,
        "covariance" => state.covariance,
        "p_c" => state.p_c,
        "p_sigma" => state.p_sigma,
        "history" => history,
        "iter_log" => iter_log,
    ), state
end

function run_optimizer(config_path::String; use_slurm::Bool=false)
    cfg = load_config(config_path)
    CURRENT_OPTIMIZER_CONFIG[] = cfg
    seed = load_json(cfg.seed_config)
    specs = build_specs(seed, cfg)
    rng = MersenneTwister(42)

    stage_outputs = Any[]
    all_history = Any[]
    state = nothing
    current_seed = deepcopy(seed)

    for stage in cfg.stages
        stage_root = joinpath(cfg.output_dir, "real_sims", stage.name)
        resume_info = stage_resume_info(stage_root)
        resume_from = resume_info === nothing ? 0 : Int(resume_info["last_iter"])
        if resume_info !== nothing
            @info "Resuming stage from artifacts" stage=stage.name resume_from=resume_from
        end
        result, state = run_stage(rng, current_seed, specs, cfg, stage, state; use_slurm=use_slurm, resume_from=resume_from)
        posterior_path = nothing
        if cfg.posterior.enabled
            posterior_path = run_nuts_from_stage(
                stage_root;
                draws=cfg.posterior.draws,
                warmup=cfg.posterior.warmup,
                max_depth=cfg.posterior.max_depth,
                step_size=cfg.posterior.step_size,
                temperature=cfg.posterior.temperature,
            )
            if posterior_path !== nothing
                posterior_state = posterior_reusable_state(posterior_path)
                state = build_state_from_reusable(
                    current_seed,
                    stage_specs(specs, cfg, stage),
                    posterior_state;
                    temporal_unlock_multiplier=1.0,
                )
                safe_save_json(
                    joinpath(stage_root, "posterior_reusable_state.json"),
                    posterior_state;
                    label="posterior_reusable_state",
                )
            end
        end
        current_seed = result["best_candidate"]
        update_stage_freeze!(cfg, stage, result["history"], specs)
        push!(stage_outputs, Dict(
            "stage" => result["stage"],
            "search_policy" => result["search_policy"],
            "fit_months" => result["fit_months"],
            "best_score" => result["best_score"],
            "top_k" => length(result["top_candidates"]),
            "sigma" => result["sigma"],
            "posterior_samples" => posterior_path,
        ))
        append!(all_history, result["history"])
        safe_save_json(joinpath(cfg.output_dir, "$(stage.name)_best_candidate.json"), result["best_candidate"]; label="stage_best_candidate")
        safe_save_json(joinpath(cfg.output_dir, "$(stage.name)_top_candidates.json"), result["top_candidates"]; label="stage_top_candidates")
        safe_save_json(joinpath(cfg.output_dir, "$(stage.name)_summary.json"), Dict(
            "stage" => result["stage"],
            "search_policy" => result["search_policy"],
            "fit_months" => result["fit_months"],
            "best_score" => result["best_score"],
            "top_k" => length(result["top_candidates"]),
            "sigma" => result["sigma"],
            "posterior_samples" => posterior_path,
            "top_candidates" => result["top_candidates"],
            "iter_log" => result["iter_log"],
        ); label="stage_summary")
    end

    safe_save_json(joinpath(cfg.output_dir, "optimizer_history.json"), all_history; label="optimizer_history")
    safe_save_json(joinpath(cfg.output_dir, "stage_summary.json"), stage_outputs; label="stage_summary")
    safe_save_json(joinpath(cfg.output_dir, "final_best_candidate.json"), current_seed; label="final_best_candidate")
    scores_by_policy = Dict{String,Vector{Float64}}()
    for stage_out in stage_outputs
        policy_name = String(stage_out["search_policy"])
        push!(get!(scores_by_policy, policy_name, Float64[]), Float64(stage_out["best_score"]))
    end
    leaderboard_entries = Any[]
    for policy_name in sort(collect(keys(scores_by_policy)))
        vals = scores_by_policy[policy_name]
        push!(leaderboard_entries, Dict(
            "search_policy" => policy_name,
            "best_score" => minimum(vals),
            "mean_stage_score" => mean(vals),
            "stages" => length(vals),
        ))
    end
    leaderboard = Dict(
        "requested_search_policy" => cfg.objective.search_policy,
        "available_policies" => candidate_search_policy_names(),
        "entries" => leaderboard_entries,
    )
    safe_save_json(joinpath(cfg.output_dir, "policy_leaderboard.json"), leaderboard; label="policy_leaderboard")
    return Dict("stage_summary" => stage_outputs, "output_dir" => cfg.output_dir, "top_k" => cfg.objective.top_k, "search_policy" => cfg.objective.search_policy)
end

function main()
    use_slurm = "--slurm" in ARGS
    config_args = filter(arg -> arg != "--slurm", ARGS)
    config_path = length(config_args) >= 1 ? config_args[1] : joinpath(dirname(@__DIR__), "optimizer_config.json")
    result = run_optimizer(config_path; use_slurm=use_slurm)
    println(JSON.json(result))
end

end
