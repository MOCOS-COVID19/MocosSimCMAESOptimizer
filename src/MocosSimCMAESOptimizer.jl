module MocosSimCMAESOptimizer

using JSON
using LinearAlgebra
using Random
using Statistics
using Dates
using HDF5
using Printf
using Serialization
using SHA

const MANAGER_ROOT = abspath(joinpath(@__DIR__, ".."))
const CURRENT_OPTIMIZER_CONFIG = Ref{Any}(nothing)
const CMA_SIGMA_MIN = 0.02
const CMA_SIGMA_MAX = 0.12
const NEW_TEMPORAL_VARIANCE = 0.04

export main, run_optimizer, run_long_horizon, run_nuts_from_archive,
       run_nuts_from_stage, posterior_reusable_state, safe_save_json,
       survivor_archive_update, archive_quality_gate, load_transfer_survivor_archive,
       persist_archive_transfer_manifest, preflight_config, create_candidate_root,
       adapter_failure, candidate_terminal_status, wait_for_iteration_outputs,
       stage_resume_info, materialize_terminal_candidate!, normalized_iteration_result,
       atomic_save_json, validate_committed_artifacts, load_immediate_predecessor_state,
       canonical_data_protocol, temporal_data_split,
       stage_data_split, negative_binomial_loglikelihood,
       run_production_smoke, validate_simulation_jld2,
       compare_smoke_manifests

struct ExternalSimConfig
    gt_dir::String
    julia_bin::String
    project_dir::String
    advanced_cli::String
    disable_compiled_modules::Bool
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
    temporal_jump_weight::Float64
    infection_extrema_weight::Float64
end

struct PosteriorConfig
    enabled::Bool
    likelihood::String
    draws::Int
    warmup::Int
    max_depth::Int
    step_size::Float64
    temperature::Float64
    transfer_covariance_inflation::Float64
    transfer_sigma_multiplier::Float64
    new_dimension_variance::Float64
    immigrant_fraction::Float64
end

struct OptimizerConfig
    seed_config::String
    output_dir::String
    monthly_days::Int
    stages::Vector{StageConfig}
    scalar_bounds::Dict{String,Tuple{Float64,Float64}}
    temporal_bounds::Dict{String,Tuple{Float64,Float64}}
    scalar_preprocessing::Dict{String,Dict{String,Any}}
    temporal_parameterization::String
    age_population_weights::Dict{String,Float64}
    validation::Dict{String,Any}
    objective::ObjectiveConfig
    external_sim::Union{Nothing,ExternalSimConfig}
    stage_freeze::Dict{String,Vector{String}}
    initial_state::Union{Nothing,Dict{String,Any}}
    posterior::PosteriorConfig
    runtime_seed::Dict{String,Any}
end

# Backwards-compatible constructor for fixture/config callers that do not
# carry the normalized runtime seed explicitly.
OptimizerConfig(seed_config, output_dir, monthly_days, stages, scalar_bounds,
                temporal_bounds, scalar_preprocessing, temporal_parameterization,
                age_population_weights, validation, objective, external_sim,
                stage_freeze, initial_state, posterior) =
    OptimizerConfig(seed_config, output_dir, monthly_days, stages, scalar_bounds,
                    temporal_bounds, scalar_preprocessing, temporal_parameterization,
                    age_population_weights, validation, objective, external_sim,
                    stage_freeze, initial_state, posterior,
                    Dict{String,Any}())

const DEFAULT_AGE_POPULATION_WEIGHTS = Dict{String,Float64}(
    "00_04" => 0.043326963479,
    "05_14" => 0.092000943853,
    "15_34" => 0.191816378028,
    "35_59" => 0.331854151940,
    "60_79" => 0.248931116037,
    "80_plus" => 0.092070446663,
)

include("posterior_sampler.jl")
include("data_protocol.jl")
include("config_preflight.jl")
include("production_smoke.jl")

struct ParamSpec
    name::String
    kind::Symbol
    length::Int
    lower::Float64
    upper::Float64
end

struct CMAState
    mean::Vector{Float64}
    sigma::Vector{Float64}
    covariance::Matrix{Float64}
    p_c::Vector{Float64}
    p_sigma::Vector{Float64}
end

CMAState(mean::Vector{Float64}, sigma::Real, covariance::Matrix{Float64}) =
    CMAState(mean, fill(Float64(sigma), length(mean)), covariance, zeros(length(mean)), zeros(length(mean)))

CMAState(mean::Vector{Float64}, sigma::Vector{Float64}, covariance::Matrix{Float64}) =
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

function full_reusable_state_from_cma(stage::StageConfig, specs_stage::Vector{ParamSpec}, state::CMAState;
                                      transition_report=nothing)
    result = Dict(
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
    transition_report === nothing || (result["transition_delta_report"] = transition_report)
    return result
end

function stage_transition_state(
    prev::CMAState,
    stage::StageConfig,
    specs_stage::Vector{ParamSpec};
    previous_specs::Union{Nothing,Vector{ParamSpec}}=nothing,
    sigma_floor::Float64=0.08,
    sigma_scale::Float64=2.0,
)
    dim = sum(spec.length for spec in specs_stage)
    old_dim = length(prev.mean)
    validate_cma_state(prev, old_dim)["valid"] ||
        throw(ArgumentError("invalid previous CMA state"))
    cov = NEW_TEMPORAL_VARIANCE .* Matrix{Float64}(I, dim, dim)
    old_names = previous_specs === nothing ?
        ["state[$i]" for i in 1:old_dim] :
        ["$(spec.name)[$i]" for spec in previous_specs for i in 1:spec.length]
    new_names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    length(unique(old_names)) == length(old_names) ||
        throw(ArgumentError("duplicate previous CMA parameter names"))
    length(unique(new_names)) == length(new_names) ||
        throw(ArgumentError("duplicate target CMA parameter names"))
    old_map = Dict(name => i for (i, name) in enumerate(old_names))
    new_map = Dict(name => i for (i, name) in enumerate(new_names))
    if size(prev.covariance, 1) == old_dim && size(prev.covariance, 2) == old_dim
        for (name_a, new_a) in new_map
            haskey(old_map, name_a) || continue
            old_a = old_map[name_a]
            for (name_b, new_b) in new_map
                haskey(old_map, name_b) || continue
                cov[new_a, new_b] = 0.9 * prev.covariance[old_a, old_map[name_b]]
            end
        end
    end
    # Preserve the posterior-informed local geometry while adding a small
    # regularization term for the next stage. New temporal dimensions start
    # from the last transferred value and receive independent uncertainty.
    mean = fill(0.5, dim)
    for (name, new_idx) in new_map
        haskey(old_map, name) || continue
        mean[new_idx] = prev.mean[old_map[name]]
    end
    # New temporal buckets inherit the tail of the same named parameter,
    # never the tail of an unrelated scalar or temporal parameter.
    for spec in specs_stage
        spec.kind == :temporal || continue
        base = spec.name
        prior = [i for (i, n) in enumerate(old_names) if startswith(n, base * "[")]
        isempty(prior) && continue
        tail = prev.mean[last(prior)]
        for i in 1:spec.length
            name = "$(base)[$i]"
            haskey(old_map, name) || (mean[new_map[name]] = tail)
        end
    end
    sigma = fill(clamp(max(0.75 * stage.sigma, sigma_floor), CMA_SIGMA_MIN, CMA_SIGMA_MAX), dim)
    for (name, new_idx) in new_map
        haskey(old_map, name) || continue
        old_idx = old_map[name]
        old_idx <= length(prev.sigma) &&
            (sigma[new_idx] = clamp(max(prev.sigma[old_idx] * sigma_scale, sigma_floor), CMA_SIGMA_MIN, CMA_SIGMA_MAX))
    end
    p_c = zeros(dim)
    p_sigma = zeros(dim)
    for (name, new_idx) in new_map
        haskey(old_map, name) || continue
        old_idx = old_map[name]
        old_idx <= length(prev.p_c) && (p_c[new_idx] = prev.p_c[old_idx])
        old_idx <= length(prev.p_sigma) && (p_sigma[new_idx] = prev.p_sigma[old_idx])
    end
    return CMAState(
        mean,
        sigma,
        cov,
        p_c,
        p_sigma,
    )
end

"""Report effective named-coordinate deltas for a transition."""
function transition_delta_report(previous::AbstractDict, current::AbstractDict;
    coordinate_space::String="effective", version::String="v1",
    limit::Float64=0.15, candidate_class::String="archive_transfer",
    coordinate_classes=nothing, policy::String="reject")
    old_names = String.(get(previous, "param_names", String[]))
    new_names = String.(get(current, "param_names", String[]))
    old_values = Float64.(get(previous, "values", get(previous, "mean", Float64[])))
    new_values = Float64.(get(current, "values", get(current, "mean", Float64[])))
    old_map = Dict(n => i for (i, n) in enumerate(old_names))
    rows = Any[]
    for (i, name) in enumerate(new_names)
        has_old = haskey(old_map, name) && old_map[name] <= length(old_values)
        old = has_old ? old_values[old_map[name]] : nothing
        new = i <= length(new_values) ? new_values[i] : NaN
        delta = has_old ? new - old : 0.0
        klass = coordinate_classes !== nothing && i <= length(coordinate_classes) ?
            String(coordinate_classes[i]) :
            (has_old ? (abs(delta) <= 1e-12 ? "locked" : "archive_transfer") : "new_dimension")
        outcome = if !has_old || abs(delta) <= limit
            "accepted"
        elseif policy == "clip"
            "clipped"
        elseif policy == "exception"
            "exception"
        else
            "rejected"
        end
        push!(rows, Dict("name" => name, "raw_value" => new, "effective_value" => new,
            "prior_value" => old, "delta" => delta, "class" => klass,
            "limit" => limit, "policy_outcome" => outcome,
            "provenance" => candidate_class))
    end
    deltas = [abs(Float64(r["delta"])) for r in rows if r["class"] == "archive_transfer"]
    overall = any(x > limit for x in deltas) ?
        (policy == "clip" ? "clipped" : policy) : "accepted"
    return Dict{String,Any}("coordinate_space" => coordinate_space, "version" => version,
        "candidate_class" => candidate_class, "coordinates" => rows,
        "max_abs_delta" => isempty(deltas) ? 0.0 : maximum(deltas),
        "norm" => isempty(deltas) ? 0.0 : norm(deltas), "limit" => limit,
        "policy_outcome" => overall)
end

"""Validate all dimensions and numerical invariants before CMA sampling."""
function validate_cma_state(state::CMAState, expected_dim::Int=length(state.mean))
    ok = length(state.mean) == expected_dim &&
         length(state.sigma) == expected_dim &&
         length(state.p_c) == expected_dim &&
         length(state.p_sigma) == expected_dim &&
         size(state.covariance) == (expected_dim, expected_dim) &&
         all(isfinite, state.mean) && all(isfinite, state.sigma) &&
         all(isfinite, state.p_c) && all(isfinite, state.p_sigma) &&
         all(isfinite, state.covariance) &&
         all(state.sigma .> 0)
    symmetric = size(state.covariance) == (expected_dim, expected_dim) &&
                isapprox(state.covariance, state.covariance'; atol=1e-10)
    psd = false
    if symmetric
        try
            psd = minimum(eigvals(Symmetric(state.covariance))) >= -1e-8
        catch
            psd = false
        end
    end
    return Dict{String,Any}("valid" => (ok && symmetric && psd),
        "dimension" => expected_dim, "mean_length" => length(state.mean),
        "sigma_length" => length(state.sigma), "covariance_shape" => collect(size(state.covariance)),
        "symmetric" => symmetric, "positive_semidefinite" => psd,
        "finite" => (all(isfinite, state.mean) && all(isfinite, state.sigma) &&
                     all(isfinite, state.p_c) && all(isfinite, state.p_sigma) &&
                     all(isfinite, state.covariance)))
end

"""Persist a complete RNG stream, rather than only its initial seed."""
function rng_snapshot(rng::AbstractRNG)
    io = IOBuffer()
    serialize(io, rng)
    return Dict{String,Any}("algorithm" => string(typeof(rng)),
        "state" => Int.(take!(io)))
end

function restore_rng(snapshot::AbstractDict)
    haskey(snapshot, "state") || throw(ArgumentError("RNG snapshot has no state"))
    bytes = UInt8.(snapshot["state"])
    io = IOBuffer(bytes)
    rng = deserialize(io)
    rng isa AbstractRNG || throw(ArgumentError("RNG snapshot is not an RNG"))
    return rng
end

function matrix_from_json(value, dim::Int)
    if value isa AbstractMatrix
        matrix = Float64.(value)
        size(matrix, 1) == dim && size(matrix, 2) == dim || error("Invalid covariance dimensions")
        return matrix
    end
    rows = collect(value)
    length(rows) == dim || error("Invalid covariance row count")
    matrix = Matrix{Float64}(undef, dim, dim)
    for i in 1:dim
        row = Float64.(rows[i])
        length(row) == dim || error("Invalid covariance column count")
        matrix[i, :] .= row
    end
    return matrix
end

function initial_state_from_config(cfg::OptimizerConfig, dim::Int, default_mean::Vector{Float64})
    cfg.initial_state === nothing && return nothing
    raw = cfg.initial_state
    mean = haskey(raw, "mean") ? Float64.(raw["mean"]) : copy(default_mean)
    sigma = if haskey(raw, "sigma")
        raw_sigma = raw["sigma"]
        raw_sigma isa AbstractVector ? Float64.(raw_sigma) : fill(Float64(raw_sigma), dim)
    else
        fill(0.3, dim)
    end
    cov = if haskey(raw, "covariance")
        matrix_from_json(raw["covariance"], dim)
    else
        Matrix{Float64}(I, dim, dim)
    end
    size(cov, 1) == dim && size(cov, 2) == dim || return nothing
    length(mean) == dim || return nothing
    length(sigma) == dim || return nothing
    p_c = haskey(raw, "p_c") ? Float64.(raw["p_c"]) : zeros(dim)
    p_sigma = haskey(raw, "p_sigma") ? Float64.(raw["p_sigma"]) : zeros(dim)
    length(p_c) == dim && length(p_sigma) == dim || return nothing
    return CMAState(mean, clamp.(sigma, CMA_SIGMA_MIN, CMA_SIGMA_MAX), cov, p_c, p_sigma)
end

function load_full_reusable_state(path::String)
    isfile(path) || return nothing
    raw = load_json(path)
    haskey(raw, "param_names") && haskey(raw, "mean") && haskey(raw, "covariance") || return nothing
    return raw
end

function build_state_from_reusable(
    seed::Dict{String,Any},
    specs_stage::Vector{ParamSpec},
    reusable::Dict{String,Any};
    sigma_floor::Float64=0.08,
    temporal_unlock_multiplier::Float64=1.0,
    covariance_inflation::Float64=1.75,
    sigma_multiplier::Float64=1.25,
    new_dimension_variance::Float64=2.0,
    rng::AbstractRNG,
)
    old_names = [String(x) for x in reusable["param_names"]]
    old_mean = Float64.(reusable["mean"])
    length(old_names) == length(old_mean) ||
        throw(ArgumentError("reusable state name/mean dimensions disagree"))
    isempty(old_names) && throw(ArgumentError("reusable state has no parameter names"))
    length(unique(old_names)) == length(old_names) ||
        throw(ArgumentError("reusable state has duplicate parameter names"))
    all(occursin(r"^.+\[[1-9][0-9]*\]$", name) for name in old_names) ||
        throw(ArgumentError("reusable state contains noncanonical parameter names"))
    all(isfinite, old_mean) || throw(ArgumentError("reusable state mean is non-finite"))
    old_cov = matrix_from_json(reusable["covariance"], length(old_mean))
    all(isfinite, old_cov) &&
        isapprox(old_cov, old_cov'; atol=1e-10) ||
        throw(ArgumentError("reusable state covariance must be finite and symmetric"))
    minimum(eigvals(Symmetric(old_cov))) >= -1e-8 ||
        throw(ArgumentError("reusable state covariance must be positive semidefinite"))
    old_sigma = if haskey(reusable, "sigma")
        raw_sigma = reusable["sigma"]
        raw_sigma isa AbstractVector ? Float64.(raw_sigma) : fill(Float64(raw_sigma), length(old_mean))
    else
        fill(0.3, length(old_mean))
    end
    length(old_sigma) == length(old_mean) && all(isfinite, old_sigma) && all(>(0), old_sigma) ||
        throw(ArgumentError("reusable state sigma dimensions or values are invalid"))

    new_names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    new_mean = initial_vector(seed, specs_stage)
    dim = length(new_names)
    new_cov = new_dimension_variance .* Matrix{Float64}(I, dim, dim)
    new_sigma = fill(sigma_floor, dim)
    old_p_c = haskey(reusable, "p_c") ? Float64.(reusable["p_c"]) : zeros(length(old_names))
    old_p_sigma = haskey(reusable, "p_sigma") ? Float64.(reusable["p_sigma"]) : zeros(length(old_names))
    length(old_p_c) == length(old_names) && all(isfinite, old_p_c) ||
        throw(ArgumentError("reusable state p_c dimensions or values are invalid"))
    length(old_p_sigma) == length(old_names) && all(isfinite, old_p_sigma) ||
        throw(ArgumentError("reusable state p_sigma dimensions or values are invalid"))
    new_p_c = zeros(dim)
    new_p_sigma = zeros(dim)
    mapped = falses(dim)
    idx_map = Dict(name => i for (i, name) in enumerate(old_names))
    kept = Int[]
    for (j, name) in enumerate(new_names)
        haskey(idx_map, name) || continue
        i = idx_map[name]
        new_mean[j] = old_mean[i]
        new_sigma[j] = clamp(max(old_sigma[i] * sigma_multiplier, sigma_floor), CMA_SIGMA_MIN, CMA_SIGMA_MAX)
        mapped[j] = true
        i <= length(old_p_c) && (new_p_c[j] = old_p_c[i])
        i <= length(old_p_sigma) && (new_p_sigma[j] = old_p_sigma[i])
        push!(kept, i)
    end
    for (a, name_a) in enumerate(new_names), (b, name_b) in enumerate(new_names)
        haskey(idx_map, name_a) && haskey(idx_map, name_b) || continue
        ia = idx_map[name_a]
        ib = idx_map[name_b]
        new_cov[a, b] = covariance_inflation * old_cov[ia, ib]
    end
    # Phase 1 temporal unlock on resume:
    # keep scalar warm-starts stable, but loosen temporal params so the search can
    # escape minima inherited from shorter horizons / older resumed states.
    temporal_mean_jitter = 0.0
    temporal_covariance_inflation = 2.5 * temporal_unlock_multiplier
    idx = 1
    for spec in specs_stage
        if spec.kind == :temporal
            for local_idx in 1:spec.length
                pos = idx + local_idx - 1
                # Keep newly introduced dimensions at their seed values and
                # give them broad prior variance; only loosen transferred
                # temporal dimensions around their posterior mean.
                frac = spec.length <= 1 ? 1.0 : (local_idx - 1) / (spec.length - 1)
                if mapped[pos]
                    jitter_scale = temporal_mean_jitter * (0.5 + frac)
                    new_mean[pos] = clamp(
                        new_mean[pos] + jitter_scale * (2rand(rng) - 1),
                        spec.lower,
                        spec.upper,
                    )
                    new_cov[pos, pos] = min(
                        max(new_cov[pos, pos] * temporal_covariance_inflation * (1.0 + frac), 1e-6),
                        NEW_TEMPORAL_VARIANCE,
                    )
                else
                    new_cov[pos, pos] = min(max(new_cov[pos, pos], 1e-6), NEW_TEMPORAL_VARIANCE)
                end
            end
        end
        idx += spec.length
    end
    return CMAState(
        new_mean,
        new_sigma,
        new_cov,
        new_p_c,
        new_p_sigma,
    )
end

function temporal_unlock_from_top_candidates!(state::CMAState, specs_stage::Vector{ParamSpec}, top_candidates;
                                              rng::AbstractRNG)
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
                    state.mean[pos] + jitter_scale * (2rand(rng) - 1),
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

function temporal_unlock_from_bucket_errors!(state::CMAState, specs_stage::Vector{ParamSpec}, top_candidates;
                                             rng::AbstractRNG)
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
                        state.mean[pos] + jitter_scale * (2rand(rng) - 1),
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

"""Load the only trusted source for a fresh-process stage extension.

The in-process `run_optimizer` path already carries `state` and the archive
forward.  A new Julia process has neither, so it must reconstruct both from
the immediately preceding committed stage, never from the initial seed.
"""
function load_immediate_predecessor_state(cfg::OptimizerConfig, stage::StageConfig)
    index = findfirst(s -> s.name == stage.name, cfg.stages)
    index === nothing && throw(ArgumentError("stage is not present in configured stage plan"))
    index == 1 && return nothing
    predecessor = cfg.stages[index - 1]
    root = joinpath(cfg.output_dir, "real_sims", predecessor.name)
    check = validate_committed_artifacts(root, "production-v1")
    check["valid"] || throw(ArgumentError(
        "trusted predecessor artifacts failed validation: " *
        join(String.(check["contradictions"]), "; ")))
    state_path = joinpath(root, "stage_state.json")
    reusable_path = joinpath(root, "full_reusable_state.json")
    prior_state = load_json(state_path)
    reusable = load_json(reusable_path)
    String(get(prior_state, "stage", "")) == predecessor.name ||
        throw(ArgumentError("trusted predecessor stage identity mismatch"))
    String(get(reusable, "stage", predecessor.name)) == predecessor.name ||
        throw(ArgumentError("trusted predecessor reusable stage identity mismatch"))
    Int(get(prior_state, "fit_months", predecessor.fit_months)) == predecessor.fit_months ||
        throw(ArgumentError("trusted predecessor horizon mismatch"))
    # These fields are the trusted handoff, not optional annotations.  A
    # production predecessor which omits one can otherwise be silently
    # rebuilt from the seed or scalar best candidate.
    historical = get(prior_state, "historical_trajectory", nothing)
    reusable_historical = get(reusable, "historical_trajectory", nothing)
    historical isa AbstractDict && reusable_historical isa AbstractDict ||
        throw(ArgumentError("trusted predecessor historical trajectory missing"))
    historical == reusable_historical ||
        throw(ArgumentError("trusted predecessor historical trajectory mismatch"))
    haskey(historical, "identity") && haskey(historical, "prefix_values") ||
        throw(ArgumentError("trusted predecessor historical trajectory incomplete"))
    prefix_hash = get(prior_state, "prefix_hash", nothing)
    reusable_prefix_hash = get(reusable, "prefix_hash", nothing)
    prefix_hash isa AbstractString && !isempty(prefix_hash) &&
        reusable_prefix_hash isa AbstractString && prefix_hash == reusable_prefix_hash ||
        throw(ArgumentError("trusted predecessor prefix hash missing or mismatched"))
    bytes2hex(SHA.sha256(JSON.json(historical["prefix_values"]))) == prefix_hash ||
        throw(ArgumentError("trusted predecessor prefix hash content mismatch"))
    locked = get(prior_state, "locked_intervals", nothing)
    reusable_locked = get(reusable, "locked_intervals", nothing)
    locked isa AbstractVector && reusable_locked isa AbstractVector &&
        locked == reusable_locked ||
        throw(ArgumentError("trusted predecessor prefix locks missing or mismatched"))
    state_cma = get(prior_state, "cma_state", nothing)
    reusable_cma = get(reusable, "cma_state", nothing)
    state_cma isa AbstractDict && reusable_cma isa AbstractDict ||
        throw(ArgumentError("trusted predecessor nested CMA state missing"))
    state_cma == reusable_cma ||
        throw(ArgumentError("trusted predecessor nested CMA state mismatch"))
    all(haskey(state_cma, key) for key in
        ("parameter_names", "mean", "sigma", "covariance", "p_c", "p_sigma")) ||
        throw(ArgumentError("trusted predecessor nested CMA state incomplete"))
    names = get(reusable, "param_names", Any[])
    mean = get(reusable, "mean", Any[])
    length(names) == length(mean) || throw(ArgumentError(
        "trusted predecessor reusable state dimensions disagree"))
    # The canonical manifest is authoritative for transfer ordering and
    # archive contents.  Do not accept a sibling archive or a re-selection.
    archive = load_transfer_survivor_archive(
        joinpath(cfg.output_dir, "real_sims"), stage.name;
        predecessor_stage=predecessor.name,
        expected_fit_months=predecessor.fit_months,
        expected_manifest_path=joinpath(root, "archive_transfer_manifest.json"),
        stage_order=[s.name for s in cfg.stages])
    archive_ids = [String(get(x, "candidate", get(x, "id", ""))) for x in archive]
    prior_ids = get(prior_state, "archive_ids", nothing)
    reusable_ids = get(reusable, "archive_ids", nothing)
    selected_ids = get(reusable, "selected_archive_ids", nothing)
    prior_ids isa AbstractVector && reusable_ids isa AbstractVector &&
        selected_ids isa AbstractVector ||
        throw(ArgumentError("trusted predecessor admitted IDs missing"))
    string.(prior_ids) == archive_ids && string.(reusable_ids) == archive_ids &&
        string.(selected_ids) == archive_ids ||
        throw(ArgumentError("trusted predecessor admitted IDs mismatch"))
    lineage = get(reusable, "archive_lineage", nothing)
    lineage isa AbstractDict || throw(ArgumentError("trusted predecessor archive lineage missing"))
    String(get(lineage, "canonical_archive_path", "")) ==
        abspath(joinpath(root, "survivor_archive.json")) ||
        throw(ArgumentError("trusted predecessor archive lineage path mismatch"))
    get(lineage, "archive_ids", Any[]) == archive_ids ||
        throw(ArgumentError("trusted predecessor archive lineage IDs mismatch"))
    return Dict{String,Any}(
        "stage_state" => prior_state,
        "reusable_state" => reusable,
        "archive" => archive,
        "root" => root,
    )
end

"""Validate and join the durable artifacts of one committed iteration."""
function validate_committed_artifacts(stage_root::String, expected_schema::AbstractString)
    required = ["stage_state.json", "iter_metrics.jsonl",
        "top_candidates.json", "survivor_archive.json", "full_reusable_state.json"]
    paths = Dict(name => joinpath(stage_root, name) for name in required)
    contradictions = String[]
    for (name, path) in paths
        isfile(path) || push!(contradictions, "missing artifact: $name")
    end
    isempty(contradictions) || return Dict("valid" => false,
        "contradictions" => contradictions, "stage_root" => abspath(stage_root),
        "artifact_hashes" => Dict{String,Any}())
    stage_state = try load_json(paths["stage_state.json"]) catch
        push!(contradictions, "malformed stage_state.json"); Dict{String,Any}() end
    top = try load_json(paths["top_candidates.json"]) catch
        push!(contradictions, "malformed top_candidates.json"); Any[] end
    archive = try load_json(paths["survivor_archive.json"]) catch
        push!(contradictions, "malformed survivor_archive.json"); Any[] end
    reusable = try load_json(paths["full_reusable_state.json"]) catch
        push!(contradictions, "malformed full_reusable_state.json"); Dict{String,Any}() end
    metrics = Any[]
    try
        for line in eachline(paths["iter_metrics.jsonl"])
            isempty(strip(line)) || push!(metrics, JSON.parse(line))
        end
    catch
        push!(contradictions, "malformed iter_metrics.jsonl")
    end
    top isa AbstractVector || push!(contradictions, "top_candidates is not an array")
    archive isa AbstractVector || push!(contradictions, "survivor_archive is not an array")
    stage = String(get(stage_state, "stage", ""))
    iteration = Int(get(stage_state, "iteration", 0))
    names = get(stage_state, "param_names", Any[])
    commit_path = joinpath(stage_root, "iter_$(iteration)", "iteration_commit.json")
    committed = false
    if isfile(commit_path)
        commit = try load_json(commit_path) catch; Dict{String,Any}() end
        committed = get(commit, "status", "") == "committed" &&
            String(get(commit, "stage", "")) == stage &&
            Int(get(commit, "iteration", -1)) == iteration
        # A commit is only trusted when its hash manifest is self-consistent.
        # Older hand-built fixtures may omit the manifest; production commits
        # are never allowed to use that compatibility path.
        if haskey(commit, "artifact_hashes")
            hashes = commit["artifact_hashes"]
            hashes isa AbstractDict || push!(contradictions, "artifact hash manifest is not an object")
            if hashes isa AbstractDict
                keys_manifest = sort(String.(collect(keys(hashes))))
                haskey(commit, "artifact_key_set") ||
                    push!(contradictions, "committed artifact key set is missing")
                expected_keys = haskey(commit, "artifact_key_set") ?
                    sort(String.(commit["artifact_key_set"])) : String[]
                keys_manifest == expected_keys ||
                    push!(contradictions, "artifact hash manifest key set mismatch")
                production_keys = [
                    "stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
                    "survivor_archive.json", "survivor_archive_summary.json",
                    "full_reusable_state.json", "archive_transfer_manifest.json",
                    joinpath("iter_$(iteration)", "candidate_list.txt"),
                    joinpath("iter_$(iteration)", "cma_sampling_state.json"),
                    joinpath("iter_$(iteration)", "top_candidates.json"),
                ]
                fixture_keys = sort(collect(keys(paths)))
                schema = get(commit, "schema_version", nothing)
                expected_schema in ("production-v1", "fixture-v1") ||
                    push!(contradictions, "unknown expected committed schema context")
                schema isa AbstractString ||
                    push!(contradictions, "committed schema_version is missing")
                schema in ("production-v1", "fixture-v1") ||
                    push!(contradictions, "unknown committed schema_version")
                schema == expected_schema ||
                    push!(contradictions, "committed schema_version does not match expected context")
                expected_exact = schema == "production-v1" ? sort(production_keys) :
                    schema == "fixture-v1" ? fixture_keys : String[]
                keys_manifest == expected_exact ||
                    push!(contradictions, "artifact hash manifest has missing or extra artifact key")
                all(isfile(joinpath(stage_root, relative)) for relative in keys_manifest) ||
                    push!(contradictions, "artifact hash manifest contains unknown artifact key")
                if haskey(commit, "artifact_hash_manifest")
                    digest = bytes2hex(SHA.sha256(JSON.json(hashes)))
                    String(commit["artifact_hash_manifest"]) == digest ||
                        push!(contradictions, "artifact hash manifest integrity mismatch")
                end
                haskey(commit, "artifact_integrity_digest") ||
                    push!(contradictions, "committed artifact integrity digest is missing")
                if haskey(commit, "artifact_integrity_digest")
                    digest = bytes2hex(SHA.sha256(JSON.json(hashes)))
                    String(commit["artifact_integrity_digest"]) == digest ||
                        push!(contradictions, "committed artifact integrity digest mismatch")
                end
                for (relative, expected) in hashes
                    artifact = joinpath(stage_root, String(relative))
                    isfile(artifact) || push!(contradictions, "missing committed artifact: $relative")
                    isfile(artifact) && bytes2hex(SHA.sha256(read(artifact))) != String(expected) &&
                        push!(contradictions, "committed artifact hash mismatch: $relative")
                end
            end
        elseif committed
            # This is the production path, so a new commit without an exact
            # manifest must fail closed rather than silently downgrade to the
            # legacy fixture behavior.
            push!(contradictions, "committed iteration has no artifact hash manifest")
        end
    end
    committed || push!(contradictions, "missing or mismatched committed iteration manifest")
    identities = Tuple{String,Int,Int}[]
    status_by_id = Dict{Tuple{String,Int,Int},String}()
    score_by_id = Dict{Tuple{String,Int,Int},Float64}()
    candidate_int(value) = value isa Integer ? Int(value) : parse(Int, string(value))
    for row in metrics
        row isa AbstractDict || (push!(contradictions, "non-object iter_metrics row"); continue)
        identity = (String(get(row, "stage", "")), Int(get(row, "iteration", 0)),
            candidate_int(get(row, "candidate", 0)))
        identity in identities && push!(contradictions, "duplicate iter_metrics identity: $identity")
        push!(identities, identity)
        status_by_id[identity] = String(get(row, "status", ""))
        score_by_id[identity] = Float64(get(row, "score", Inf))
        identity[1] == stage || push!(contradictions, "iter_metrics stage mismatch: $identity")
    end
    function check_entry(entry, label)
        entry isa AbstractDict || (push!(contradictions, "$label is not an object"); return nothing)
        identity = (String(get(entry, "stage", "")), Int(get(entry, "iteration", 0)),
            candidate_int(get(entry, "candidate", 0)))
        identity in identities || push!(contradictions, "$label orphan identity: $identity")
        identity[1] == stage || push!(contradictions, "$label stage mismatch: $identity")
        identity[2] == iteration || push!(contradictions, "$label iteration mismatch: $identity")
        get(entry, "parameter_names", names) == names ||
            push!(contradictions, "$label parameter_names mismatch")
        haskey(status_by_id, identity) && String(get(entry, "status", "")) != status_by_id[identity] &&
            push!(contradictions, "$label status mismatch: $identity")
        haskey(score_by_id, identity) && Float64(get(entry, "score", Inf)) != score_by_id[identity] &&
            push!(contradictions, "$label score mismatch: $identity")
        identity
    end
    top_ids = Tuple{String,Int,Int}[]
    for entry in top
        id = check_entry(entry, "top_candidates")
        id === nothing || push!(top_ids, id)
    end
    archive_ids = Tuple{String,Int,Int}[]
    for entry in archive
        id = check_entry(entry, "survivor_archive")
        id === nothing || push!(archive_ids, id)
    end
    best_id = get(stage_state, "best_candidate_id", nothing)
    best_id !== nothing && !any(get(e, "candidate", 0) == best_id for e in top) &&
        push!(contradictions, "best_candidate_id is absent from top_candidates")
    haskey(reusable, "stage") && String(reusable["stage"]) != stage &&
        push!(contradictions, "full_reusable_state stage mismatch")
    admitted = get(reusable, "archive_ids", Any[])
    admitted isa AbstractVector || push!(contradictions, "reusable state archive_ids is not an array")
    for id in admitted
        any(string(get(e, "candidate", get(e, "id", ""))) == string(id) for e in archive) ||
            push!(contradictions, "reusable state references non-admitted archive id: $id")
    end
    counts = Dict("completed" => 0, "failed" => 0, "skipped" => 0, "pending" => 0)
    for row in metrics
        status = String(get(row, "status", ""))
        haskey(counts, status) ? (counts[status] += 1) :
            push!(contradictions, "unknown candidate status: $status")
    end
    for status in ("completed", "failed", "skipped", "pending")
        field = "$(status)_count"
        haskey(stage_state, field) && Int(stage_state[field]) != counts[status] &&
            push!(contradictions, "stage_state $field mismatch")
    end
    if best_id !== nothing && haskey(score_by_id, (stage, iteration, candidate_int(best_id))) &&
       haskey(stage_state, "best_score") &&
       Float64(stage_state["best_score"]) != score_by_id[(stage, iteration, candidate_int(best_id))]
        push!(contradictions, "stage_state best_score mismatch")
    end
    haskey(reusable, "param_names") && reusable["param_names"] != names &&
        push!(contradictions, "full_reusable_state parameter_names mismatch")
    hashes = Dict{String,Any}(name => bytes2hex(SHA.sha256(read(path)))
                              for (name, path) in paths)
    return Dict("valid" => isempty(contradictions), "contradictions" => contradictions,
        "stage_root" => abspath(stage_root), "stage" => stage, "iteration" => iteration,
        "committed" => committed, "status_counts" => counts,
        "jsonl_record_count" => length(metrics), "candidate_ids" => identities,
        "top_candidate_ids" => top_ids, "archive_candidate_ids" => archive_ids,
        "artifact_hashes" => hashes)
end

# Keep the context requirement explicit while allowing callers that prefer a
# named argument to state the same contract.
function validate_committed_artifacts(stage_root::String; expected_schema::AbstractString)
    validate_committed_artifacts(stage_root, expected_schema)
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

const SURVIVOR_ARCHIVE_SIZE = 200
const SURVIVOR_MIN_DISTANCE = 0.03
const SURVIVOR_SCORE_MAD_MULTIPLIER = 2.0
const SURVIVOR_RELATIVE_SCORE_FLOOR = 0.05

function archive_metric(entry::AbstractDict, name::String)
    metrics = get(entry, "metrics", Dict{String,Any}())
    metrics = metrics isa AbstractDict && haskey(metrics, "metrics") ? metrics["metrics"] : metrics
    metrics isa AbstractDict || return Inf
    value = get(metrics, name, Inf)
    return value isa Number ? Float64(value) : Inf
end

function archive_objectives(entry::AbstractDict)
    return Float64[
        Float64(get(entry, "score", Inf)),
        archive_metric(entry, "weekly_control_score"),
        archive_metric(entry, "daily_detections_cumulative"),
        archive_metric(entry, "daily_age_05_14_detections"),
        archive_metric(entry, "temporal_jump_penalty") +
            archive_metric(entry, "infection_extrema_penalty"),
    ]
end

function objective_dominates(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    comparable = false
    strictly_better = false
    for (x, y) in zip(a, b)
        isfinite(x) && isfinite(y) || continue
        comparable = true
        x <= y || return false
        x < y && (strictly_better = true)
    end
    return comparable && strictly_better
end

function archive_parameter_distance(a::AbstractDict, b::AbstractDict)
    va = get(a, "evaluated_vector", nothing)
    vb = get(b, "evaluated_vector", nothing)
    va isa AbstractVector && vb isa AbstractVector || return Inf
    length(va) == length(vb) || return Inf
    return norm(Float64.(va) - Float64.(vb)) / sqrt(max(length(va), 1))
end

function _archive_normalized_distance(a::AbstractDict, b::AbstractDict, bounds)
    va = get(a, "evaluated_vector", nothing)
    vb = get(b, "evaluated_vector", nothing)
    va isa AbstractVector && vb isa AbstractVector || return Inf
    length(va) == length(vb) || return Inf
    xa = try Float64.(va) catch; return Inf end
    xb = try Float64.(vb) catch; return Inf end
    all(isfinite, xa) && all(isfinite, xb) || return Inf
    scales = if bounds !== nothing && length(bounds) == length(xa)
        [max(Float64(x[2]) - Float64(x[1]), eps()) for x in bounds]
    else
        # Candidate-local normalization avoids a large-scale coordinate
        # dominating diversity when no parameter bounds were supplied.
        fill(1.0, length(xa))
    end
    return norm((xa .- xb) ./ scales) / sqrt(max(length(xa), 1))
end

function _archive_rejection_reason(entry::AbstractDict; current_stage=nothing, current_fit_months=nothing)
    status = String(get(entry, "status", ""))
    current_stage !== nothing && status != "completed" && return "status"
    current_stage !== nothing && String(get(entry, "stage", "")) != String(current_stage) && return "stage"
    if current_fit_months !== nothing
        horizon = get(entry, "fit_months", get(entry, "scoring_horizon_months", nothing))
        (horizon === nothing || Int(horizon) != Int(current_fit_months)) && return "horizon"
    end
    score = try Float64(get(entry, "score", Inf)) catch; Inf end
    vector = get(entry, "evaluated_vector", nothing)
    (!isfinite(score) || !(vector isa AbstractVector)) && return "nonfinite"
    values = try Float64.(vector) catch; Float64[] end
    all(isfinite, values) || return "nonfinite"
    return nothing
end

function _archive_quality_band(scores::Vector{Float64}; mad_multiplier=SURVIVOR_SCORE_MAD_MULTIPLIER)
    best = minimum(scores)
    med = median(scores)
    mad = median(abs.(scores .- med))
    floor = SURVIVOR_RELATIVE_SCORE_FLOOR * max(abs(best), 1.0)
    scale = max(mad, floor, 1e-8)
    return Dict{String,Any}(
        "best" => best, "median" => med, "mad" => mad,
        "relative_floor" => floor, "scale" => scale,
        "multiplier" => mad_multiplier,
        "threshold" => best + mad_multiplier * scale,
        "reason" => mad > floor ? "mad" : "relative_floor",
    )
end

"""
    survivor_archive_update(existing, entries; ...)

Build the reusable archive product. `max_size` is only a technical cap;
`target_size` is the adaptive nominal target. With `return_report=true`, a
durable-policy-shaped report is returned instead of only the selected vector.
"""
function survivor_archive_update(existing, entries;
    max_size::Int=SURVIVOR_ARCHIVE_SIZE,
    target_size::Int=40,
    min_distance::Float64=SURVIVOR_MIN_DISTANCE,
    parameter_bounds=nothing,
    current_stage=nothing,
    current_fit_months=nothing,
    return_report::Bool=false)
    max_size > 0 || throw(ArgumentError("archive cap must be positive"))
    pool = Any[]
    rejected = Dict{String,Int}()
    seen = Set{String}()
    for entry in vcat(collect(existing), collect(entries))
        entry isa AbstractDict || continue
        reason = _archive_rejection_reason(entry;
            current_stage=current_stage, current_fit_months=current_fit_months)
        reason !== nothing && (rejected[reason] = get(rejected, reason, 0) + 1; continue)
        id = string(get(entry, "candidate", get(entry, "id", "")))
        if !isempty(id) && id in seen
            rejected["duplicate"] = get(rejected, "duplicate", 0) + 1
            continue
        end
        !isempty(id) && push!(seen, id)
        push!(pool, deepcopy(entry))
    end
    isempty(pool) && return return_report ? Dict{String,Any}(
        "archive" => Any[], "archive_count" => 0, "configured_target" => target_size,
        "technical_cap" => max_size, "quality_band" => nothing,
        "rejected_counts" => rejected, "adaptive_target_status" => "constrained") : Any[]
    sort!(pool, by = x -> Float64(x["score"]))

    scores = Float64[Float64(entry["score"]) for entry in pool]
    quality_band = _archive_quality_band(scores)
    score_threshold = quality_band["threshold"]
    quality_pool = [entry for entry in pool if Float64(entry["score"]) <= score_threshold]
    distance_bounds = parameter_bounds
    if distance_bounds === nothing && !isempty(quality_pool)
        vectors = [try Float64.(x["evaluated_vector"]) catch; Float64[] end for x in quality_pool]
        if !isempty(vectors) && all(v -> length(v) == length(first(vectors)), vectors)
            distance_bounds = [(minimum(v[i] for v in vectors), maximum(v[i] for v in vectors))
                               for i in eachindex(first(vectors))]
        end
    end

    objectives = [archive_objectives(entry) for entry in quality_pool]
    pareto = Any[]
    for (i, entry) in enumerate(quality_pool)
        any(
            j != i && objective_dominates(objectives[j], objectives[i])
            for j in eachindex(quality_pool)
        ) && continue
        push!(pareto, entry)
    end
    isempty(pareto) && (pareto = [first(quality_pool)])
    sort!(pareto, by = x -> (Float64(x["score"]), string(get(x, "candidate", ""))))

    selected = Any[]
    desired = min(max_size, max(target_size, 1))
    for entry in pareto
        any(_archive_normalized_distance(entry, other, distance_bounds) < min_distance for other in selected) && continue
        push!(selected, entry)
        length(selected) >= desired && break
    end
    if length(selected) < desired
        for entry in quality_pool
            any(selected_entry === entry for selected_entry in selected) && continue
            any(_archive_normalized_distance(entry, other, distance_bounds) < min_distance for other in selected) && continue
            push!(selected, entry)
            length(selected) >= desired && break
        end
    end
    constrained = length(quality_pool) < target_size
    for entry in selected
        if current_stage !== nothing
            entry["admission_reason"] = "current_stage_quality_and_diversity"
            entry["quality_threshold"] = score_threshold
            entry["archive_provenance"] = get(entry, "provenance", Dict{String,Any}())
            entry["requested_horizon"] = get(entry, "requested_horizon", current_fit_months)
            entry["effective_scoring_horizon"] = get(entry, "effective_scoring_horizon", current_fit_months)
            entry["transition_delta_report"] = get(entry, "transition_delta_report",
                Dict{String,Any}("status" => "not_available", "candidate_class" => "unknown"))
        end
    end
    pairwise = Float64[]
    for i in eachindex(selected)
        for j in (i + 1):length(selected)
            i < j && push!(pairwise, _archive_normalized_distance(selected[i], selected[j], distance_bounds))
        end
    end
    report = Dict{String,Any}(
        "archive" => selected, "archive_count" => length(selected),
        "configured_target" => target_size, "technical_cap" => max_size,
        "adaptive_target_status" => constrained ? "constrained" : "met",
        "quality_band" => quality_band, "quality_pool_count" => length(quality_pool),
        "pareto_count" => length(pareto), "min_parameter_distance" => min_distance,
        "diversity" => Dict{String,Any}(
            "pairwise_count" => length(pairwise),
            "minimum_distance" => isempty(pairwise) ? 0.0 : minimum(pairwise),
            "mean_distance" => isempty(pairwise) ? 0.0 : mean(pairwise),
        ),
        "rejected_counts" => rejected,
        "diversity_policy" => "normalized_parameter_distance",
        "scalar_best" => first(pool),
    )
    return return_report ? report : selected
end

function archive_quality_gate(archive; current_stage=nothing, current_fit_months=nothing,
    current_best_score=Inf, target_size::Int=40, minimum_size::Int=1,
    quality_band=nothing, diversity_passed=nothing,
    quality_band_constrained::Bool=false, allow_reduced_archive::Bool=true,
    current_quality_band=nothing, current_minimum_size=nothing,
    current_diversity_passed=nothing)
    quality_band = current_quality_band === nothing ? quality_band : current_quality_band
    minimum_size = current_minimum_size === nothing ? minimum_size : Int(current_minimum_size)
    diversity_passed = current_diversity_passed === nothing ? diversity_passed : current_diversity_passed
    values = archive isa AbstractVector ? collect(archive) : Any[]
    valid = [x for x in values if x isa AbstractDict &&
        _archive_rejection_reason(x; current_stage=current_stage,
            current_fit_months=current_fit_months) === nothing]
    best = isempty(valid) ? Inf : minimum(Float64(x["score"]) for x in valid)
    external_inputs_ok = quality_band isa AbstractDict &&
        haskey(quality_band, "threshold") &&
        current_minimum_size !== nothing &&
        current_diversity_passed !== nothing
    threshold = quality_band isa AbstractDict ? get(quality_band, "threshold", Inf) : Inf
    threshold = try Float64(threshold) catch; Inf end
    quality_ok = external_inputs_ok && isfinite(Float64(current_best_score)) && isfinite(threshold) &&
        isfinite(best) && best <= threshold &&
        Float64(current_best_score) <= threshold &&
        all(try isfinite(Float64(x["score"])) && Float64(x["score"]) <= threshold
            catch; false end for x in valid)
    count_ok = length(valid) >= minimum_size &&
        (length(valid) >= target_size || (quality_band_constrained && allow_reduced_archive))
    diversity_ok = diversity_passed === nothing ? !isempty(valid) : Bool(diversity_passed)
    passed = quality_ok && count_ok && diversity_ok
    reason = passed ? "current_objective_quality_and_archive_eligible" :
        (!external_inputs_ok ? "missing_external_quality_band" :
         (!isfinite(Float64(current_best_score)) || Float64(current_best_score) > threshold ?
          "current_objective_quality_failed" :
         !quality_ok ? "quality_band_failed" :
         (!count_ok ? "insufficient_admitted_count" : "insufficient_diversity")))
    return Dict{String,Any}(
        "status" => passed ? "passed" : "blocked", "next_stage_created" => false,
        "current_stage" => current_stage, "current_fit_months" => current_fit_months,
        "current_best_score" => current_best_score, "archive_best_score" => best,
        "effective_quality_threshold" => threshold,
        "configured_target" => target_size, "minimum_count" => minimum_size,
        "admitted_count" => length(valid), "quality_band" => quality_band,
        "diversity_passed" => diversity_ok, "refusal_reason" => reason,
        "reduced_archive_policy" => "allow_only_when_quality_band_constrained",
        "quality_band_constrained" => quality_band_constrained,
    )
end

function archive_vector_for_stage(entry::AbstractDict, specs_stage::Vector{ParamSpec}, fallback::Vector{Float64})
    old_names = [String(x) for x in get(entry, "parameter_names", String[])]
    old_vector = get(entry, "evaluated_vector", nothing)
    old_vector isa AbstractVector || return copy(fallback)
    old_map = Dict(name => i for (i, name) in enumerate(old_names))
    names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    vector = copy(fallback)
    for (j, name) in enumerate(names)
        haskey(old_map, name) || continue
        i = old_map[name]
        i <= length(old_vector) && (vector[j] = Float64(old_vector[i]))
    end
    return vector
end

function _archive_payload_hash(values)
    io = IOBuffer()
    JSON.print(io, values)
    return bytes2hex(sha256(take!(io)))
end

function persist_archive_transfer_manifest(stage_root::String, archive;
                                           archive_path::String=joinpath(stage_root, "survivor_archive.json"),
                                           stage::Union{Nothing,String}=nothing,
                                           fit_months::Union{Nothing,Int}=nothing)
    values = archive isa AbstractVector ? collect(archive) : Any[]
    ids = [string(get(x, "candidate", get(x, "id", ""))) for x in values if x isa AbstractDict]
    payload_hash = _archive_payload_hash(values)
    source = stage === nothing ? basename(stage_root) : stage
    manifest = Dict{String,Any}(
        "schema_version" => "archive-transfer-v2",
        "canonical_archive_path" => abspath(archive_path),
        "source_stage" => source,
        "source_fit_months" => fit_months,
        "horizon" => fit_months,
        "archive_id" => string(source, ":", fit_months === nothing ? "unknown" : fit_months, ":", payload_hash),
        "archive_hash" => payload_hash,
        "admitted_ids" => ids,
        "admitted_order" => ids,
        "archive_count" => length(ids),
    )
    path = joinpath(stage_root, "archive_transfer_manifest.json")
    safe_save_json(path, manifest; label="archive_transfer_manifest")
    return path
end

function load_transfer_survivor_archive(output_dir::String, current_stage::String;
                                        predecessor_stage::Union{Nothing,String}=nothing,
                                        expected_fit_months::Union{Nothing,Int}=nothing,
                                        expected_manifest_path::Union{Nothing,String}=nothing,
                                        expected_archive_id::Union{Nothing,String}=nothing,
                                        stage_order=nothing,
                                        configuration=nothing,
                                        return_evidence::Bool=false)
    reject(reason; details=Dict{String,Any}()) = begin
        evidence = Dict{String,Any}(
            "status" => "rejected",
            "failure_class" => String(reason),
            "current_stage" => current_stage,
            "predecessor_stage" => predecessor_stage,
            "expected_fit_months" => expected_fit_months,
        )
        merge!(evidence, details)
        return return_evidence ? evidence : Any[]
    end
    # This is an untrusted file boundary.  Every conversion and consistency
    # check is deliberately inside one rejection boundary: malformed JSON
    # must never escape as a MethodError/ArgumentError/TypeError.
    try
        # A caller-selected predecessor is not authoritative.  The direct API
        # requires either the normalized stage order or a configuration that
        # contains it, so sibling archives cannot be selected by identity.
        if stage_order === nothing && configuration !== nothing
            stage_order = if configuration isa AbstractDict
                configured_order = get(configuration, "stage_order",
                    get(configuration, :stage_order, nothing))
                configured_order === nothing && haskey(configuration, "stages") &&
                    (configured_order = [
                        s isa AbstractDict ? get(s, "name", get(s, :name, "")) :
                        (hasproperty(s, :name) ? getproperty(s, :name) : "")
                        for s in configuration["stages"]])
                configured_order
            elseif hasproperty(configuration, :stage_order)
                getproperty(configuration, :stage_order)
            elseif hasproperty(configuration, :stages)
                stages = getproperty(configuration, :stages)
                [hasproperty(s, :name) ? getproperty(s, :name) :
                    (s isa AbstractDict ? get(s, "name", "") : "") for s in stages]
            else
                nothing
            end
        end
        stage_order === nothing && return reject("missing_authoritative_stage_order")
        source_stage = begin
            order = try collect(stage_order) catch; return reject("invalid_authoritative_stage_order") end
            all(x -> x isa AbstractString && !isempty(x), order) ||
                return reject("invalid_authoritative_stage_order")
            names = String.(order)
            length(unique(names)) == length(names) ||
                return reject("invalid_authoritative_stage_order")
            current_stage in names || return reject("current_stage_not_in_authoritative_order")
            i = findfirst(==(current_stage), names)
            i === nothing && return reject("current_stage_not_in_authoritative_order")
            i <= 1 && return reject("current_stage_has_no_predecessor")
            names[i - 1]
        end
        predecessor_stage !== nothing && String(predecessor_stage) != source_stage &&
            return reject("predecessor_stage_not_immediate_predecessor";
                details=Dict("derived_predecessor_stage" => source_stage))
        # Only the immediate, canonical predecessor is trusted.  Searching
        # sibling stages can silently mix incompatible horizons/provenance.
        stage_root = joinpath(output_dir, source_stage)
        path = joinpath(stage_root, "survivor_archive.json")
        isfile(path) || return reject("missing_archive")
        manifest_path = joinpath(stage_root, "archive_transfer_manifest.json")
        isfile(manifest_path) || return reject("missing_transfer_manifest")
        expected_manifest_path !== nothing &&
            abspath(expected_manifest_path) != abspath(manifest_path) &&
            return reject("manifest_path_mismatch")
        manifest = load_json(manifest_path)
        manifest isa AbstractDict || return reject("malformed_transfer_manifest")

        required = ("schema_version", "canonical_archive_path", "source_stage",
            "source_fit_months", "horizon", "archive_id", "archive_hash",
            "admitted_ids", "admitted_order", "archive_count")
        for key in required
            haskey(manifest, key) || return reject(
                key == "schema_version" ? "schema_version_missing" :
                "missing_manifest_metadata";
                details=Dict("missing_field" => key))
        end
        manifest["schema_version"] isa AbstractString ||
            return reject("schema_version_missing")
        manifest["schema_version"] == "archive-transfer-v2" ||
            return reject("schema_version_mismatch")
        manifest["canonical_archive_path"] isa AbstractString ||
            return reject("canonical_archive_path_invalid")
        manifest["source_stage"] isa AbstractString ||
            return reject("source_stage_invalid")
        manifest["archive_id"] isa AbstractString ||
            return reject("archive_id_invalid")
        manifest["archive_hash"] isa AbstractString ||
            return reject("archive_hash_invalid")
        manifest["source_stage"] == source_stage ||
            return reject("source_stage_not_immediate_predecessor";
                details=Dict("derived_predecessor_stage" => source_stage))
        manifest["canonical_archive_path"] == abspath(path) ||
            return reject("canonical_archive_path_mismatch")
        # JSON booleans are not valid integer counts/horizons.
        isint(x) = x isa Integer && !(x isa Bool)
        isint(manifest["source_fit_months"]) || return reject("source_fit_months_invalid")
        isint(manifest["horizon"]) || return reject("horizon_invalid")
        isint(manifest["archive_count"]) || return reject("archive_count_invalid")
        manifest["source_fit_months"] == manifest["horizon"] ||
            return reject("horizon_mismatch")
        expected_fit_months !== nothing &&
            manifest["source_fit_months"] != expected_fit_months &&
            return reject("expected_horizon_mismatch")
        manifest["admitted_ids"] isa AbstractVector || return reject("admitted_ids_invalid")
        manifest["admitted_order"] isa AbstractVector || return reject("admitted_order_invalid")
        all(x -> x isa AbstractString && !isempty(x), manifest["admitted_ids"]) ||
            return reject("admitted_ids_invalid")
        all(x -> x isa AbstractString && !isempty(x), manifest["admitted_order"]) ||
            return reject("admitted_order_invalid")
        admitted_ids = String.(manifest["admitted_ids"])
        admitted_order = String.(manifest["admitted_order"])
        admitted_ids == admitted_order || return reject("admitted_order_mismatch")
        length(unique(admitted_ids)) == length(admitted_ids) ||
            return reject("duplicate_admitted_ids")

        values = load_json(path)
        values isa AbstractVector || return reject("malformed_archive")
        all(x -> x isa AbstractDict, values) || return reject("malformed_archive")
        payload_ids = String[]
        for x in values
            haskey(x, "candidate") || haskey(x, "id") || return reject("archive_id_missing")
            id = haskey(x, "candidate") ? x["candidate"] : x["id"]
            (id isa AbstractString || id isa Integer) || return reject("archive_id_invalid")
            id isa AbstractString && isempty(id) && return reject("archive_id_invalid")
            push!(payload_ids, String(id))
        end
        payload_ids == admitted_ids || return reject("archive_payload_order_mismatch")
        manifest["archive_count"] == length(admitted_ids) ||
            return reject("archive_count_mismatch")
        length(values) == manifest["archive_count"] || return reject("archive_count_mismatch")
        payload_hash = _archive_payload_hash(values)
        manifest["archive_hash"] == payload_hash || return reject("archive_hash_mismatch")
        archive_id = string(source_stage, ":", manifest["horizon"], ":", payload_hash)
        manifest["archive_id"] == archive_id || return reject("archive_id_mismatch")
        expected_archive_id !== nothing && expected_archive_id != archive_id &&
            return reject("expected_archive_id_mismatch")
        all(_archive_rejection_reason(x; current_stage=source_stage,
            current_fit_months=expected_fit_months) === nothing for x in values) ||
            return reject("archive_entry_ineligible")
        return deepcopy(values)
    catch
        return reject("archive_loader_exception")
    end
end

function archive_entry_config(archive)
    archive isa AbstractVector || return nothing
    for entry in archive
        entry isa AbstractDict || continue
        cfg = get(entry, "config", nothing)
        cfg isa AbstractDict || continue
        return deepcopy(cfg)
    end
    return nothing
end

function effective_transition_report(seed::AbstractDict, cfg::AbstractDict,
                                    specs_stage::Vector{ParamSpec},
                                    source_entry=nothing;
                                    candidate_class::String="new_dimension",
                                    limit::Float64=0.15)
    names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    values = if haskey(cfg, "values")
        Float64.(cfg["values"])
    else
        collected = Float64[]
        for spec in specs_stage
            raw = try get_nested(cfg, spec.name) catch; Any[] end
            raw isa AbstractVector || (raw = [raw])
            for value in raw
                value isa Number && push!(collected, Float64(value))
            end
        end
        collected
    end
    prior = if source_entry isa AbstractDict
        Dict{String,Any}(
            "param_names" => get(source_entry, "parameter_names", String[]),
            "values" => get(source_entry, "evaluated_vector", Float64[]),
        )
    else
        Dict{String,Any}("param_names" => names, "values" => initial_vector(seed, specs_stage))
    end
    current = Dict{String,Any}("param_names" => names, "values" => values)
    source_names = String.(get(prior, "param_names", String[]))
    classes = if candidate_class == "archive_transfer"
        old_values = Float64.(get(prior, "values", Float64[]))
        old_map = Dict(name => i for (i, name) in enumerate(source_names))
        [haskey(old_map, name) && old_map[name] <= length(old_values) &&
             i <= length(values) && abs(values[i] - old_values[old_map[name]]) <= 1e-12 ?
             "locked" : (name in source_names ? "archive_transfer" : "new_dimension")
         for (i, name) in enumerate(names)]
    else
        [candidate_class for _ in names]
    end
    return transition_delta_report(prior, current;
        candidate_class=candidate_class, coordinate_classes=classes, limit=limit)
end

"""Apply transition policy before a candidate reaches scoring or admission.

The returned config is always the effective config.  Rejected transfers have
no effective candidate and callers must classify them terminally.
"""
function enforce_transition_policy(seed::AbstractDict, cfg::AbstractDict,
                                   specs_stage::Vector{ParamSpec},
                                   source_entry=nothing;
                                   candidate_class::String="new_dimension",
                                   limit::Float64=0.15,
                                   policy::String="reject")
    policy in ("reject", "clip", "exception") ||
        throw(ArgumentError("unsupported transition policy: $policy"))
    names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    values = Float64[]
    for spec in specs_stage
        raw = try get_nested(cfg, spec.name) catch; Any[] end
        raw isa AbstractVector || (raw = [raw])
        append!(values, Float64.(raw))
    end
    length(values) == length(names) ||
        throw(ArgumentError("candidate coordinate count does not match stage specification"))
    prior = source_entry isa AbstractDict ?
        Dict{String,Any}("param_names" => get(source_entry, "parameter_names", String[]),
                         "values" => get(source_entry, "evaluated_vector", Float64[])) :
        Dict{String,Any}("param_names" => names, "values" => initial_vector(seed, specs_stage))
    old_names = String.(get(prior, "param_names", String[]))
    old_values = Float64.(get(prior, "values", Float64[]))
    old_map = Dict(name => i for (i, name) in enumerate(old_names))
    effective = copy(values)
    over_limit = false
    for (i, name) in enumerate(names)
        haskey(old_map, name) && old_map[name] <= length(old_values) || continue
        delta = effective[i] - old_values[old_map[name]]
        abs(delta) <= limit && continue
        over_limit = true
        if policy == "clip"
            effective[i] = old_values[old_map[name]] + sign(delta) * limit
        end
    end
    effective_cfg = deepcopy(cfg)
    idx = 1
    for spec in specs_stage
        if spec.kind == :scalar
            set_nested!(effective_cfg, spec.name, effective[idx])
            idx += 1
        else
            current = collect(Float64.(get_nested(effective_cfg, spec.name)))
            for j in 1:min(spec.length, length(current))
                current[j] = effective[idx]
                idx += 1
            end
            set_nested!(effective_cfg, spec.name, current)
        end
    end
    report = effective_transition_report(seed, effective_cfg, specs_stage, source_entry;
        candidate_class=candidate_class, limit=limit)
    for (i, row) in enumerate(report["coordinates"])
        row["raw_value"] = values[i]
        row["effective_value"] = effective[i]
    end
    if over_limit && policy == "reject"
        report["policy_outcome"] = "reject"
        return Dict{String,Any}("status" => "rejected", "config" => nothing, "report" => report)
    elseif over_limit && policy == "exception"
        report["policy_outcome"] = "exception"
        return Dict{String,Any}("status" => "rejected", "config" => nothing, "report" => report)
    elseif over_limit && policy == "clip"
        report["policy_outcome"] = "clipped"
    end
    return Dict{String,Any}("status" => "accepted", "config" => effective_cfg, "report" => report)
end

"""Apply the transition contract to a posterior reusable state before use.

Posterior means are already in effective coordinate space, so this helper
clips/rejects the mean directly and carries the complete report into the
state.  Callers must not construct or persist the returned state on reject.
"""
function enforce_posterior_reusable_state(
    seed::AbstractDict,
    specs_stage::Vector{ParamSpec},
    posterior_state::AbstractDict,
    source_entry=nothing;
    active_months::Int=0,
    limit::Float64=0.15,
    policy::String="reject",
)
    policy in ("reject", "clip", "exception") ||
        throw(ArgumentError("unsupported transition policy: $policy"))
    state = deepcopy(posterior_state)
    names = String.(get(state, "param_names", String[]))
    mean = Float64.(get(state, "mean", Float64[]))
    expected_names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
    length(names) == length(mean) || throw(ArgumentError("posterior state name/mean dimensions disagree"))
    names == expected_names || throw(ArgumentError("posterior state coordinates do not match target stage"))

    prior = source_entry isa AbstractDict ?
        Dict{String,Any}(
            "param_names" => get(source_entry, "parameter_names", String[]),
            "values" => get(source_entry, "evaluated_vector", Float64[]),
        ) :
        Dict{String,Any}("param_names" => expected_names,
                         "values" => initial_vector(seed, specs_stage))
    prior_names = String.(get(prior, "param_names", String[]))
    prior_values = Float64.(get(prior, "values", Float64[]))
    prior_map = Dict(name => i for (i, name) in enumerate(prior_names))
    effective = copy(mean)
    over_limit = false
    for (i, name) in enumerate(names)
        haskey(prior_map, name) && prior_map[name] <= length(prior_values) || continue
        delta = effective[i] - prior_values[prior_map[name]]
        abs(delta) <= limit && continue
        over_limit = true
        policy == "clip" && (effective[i] = prior_values[prior_map[name]] + sign(delta) * limit)
    end
    current = Dict{String,Any}("param_names" => names, "values" => effective)
    classes = [
        haskey(prior_map, name) && prior_map[name] <= length(prior_values) &&
            abs(effective[i] - prior_values[prior_map[name]]) <= 1e-12 ?
            "locked" : (haskey(prior_map, name) ? "archive_transfer" : "new_dimension")
        for (i, name) in enumerate(names)
    ]
    report = transition_delta_report(prior, current;
        candidate_class=source_entry isa AbstractDict ? "archive_transfer" : "new_dimension",
        coordinate_classes=classes, limit=limit, policy=policy)
    report["raw_values"] = mean
    report["effective_values"] = effective
    report["active_months"] = active_months
    report["source_provenance"] = source_entry isa AbstractDict ?
        Dict{String,Any}(
            "archive_entry_id" => get(source_entry, "candidate", nothing),
            "source_stage" => get(source_entry, "stage", nothing),
            "source_fit_months" => get(source_entry, "fit_months", nothing),
        ) :
        Dict{String,Any}("source" => "seed_prior")
    for (i, row) in enumerate(report["coordinates"])
        row["raw_value"] = mean[i]
        row["effective_value"] = effective[i]
        row["provenance"] = report["source_provenance"]
    end
    if over_limit && policy == "reject"
        report["policy_outcome"] = "reject"
        evidence = Dict{String,Any}(
            "status" => "failed",
            "failure_class" => "posterior_transition_policy_reject",
            "simulated" => "posterior_transition_rejected",
            "active_months" => active_months,
            "transition_delta_report" => report,
        )
        return Dict{String,Any}("status" => "rejected", "state" => nothing,
                                "report" => report, "terminal_evidence" => evidence)
    elseif over_limit && policy == "exception"
        report["policy_outcome"] = "exception"
        evidence = Dict{String,Any}(
            "status" => "failed",
            "failure_class" => "posterior_transition_policy_exception",
            "simulated" => "posterior_transition_rejected",
            "active_months" => active_months,
            "transition_delta_report" => report,
        )
        return Dict{String,Any}("status" => "rejected", "state" => nothing,
                                "report" => report, "terminal_evidence" => evidence)
    elseif over_limit && policy == "clip"
        report["policy_outcome"] = "clipped"
    end
    state["mean"] = effective
    state["transition_delta_report"] = report
    state["transition_policy_outcome"] = report["policy_outcome"]
    return Dict{String,Any}("status" => "accepted", "state" => state, "report" => report,
                            "terminal_evidence" => nothing)
end

"""
    candidate_terminal_status(candidate_dir)

Return one conservative terminal classification for a candidate directory.
Conflicting markers are terminal failures, never successful work.  This is
shared by local and collected execution so ranking cannot infer completion
from directory existence or from a finite-looking artifact alone.
"""
function candidate_terminal_status(candidate_dir::String)
    done = isfile(joinpath(candidate_dir, "done.ok"))
    failed = isfile(joinpath(candidate_dir, "failed.ok"))
    skipped = isfile(joinpath(candidate_dir, "skipped.ok"))
    markers = count(identity, (done, failed, skipped))
    if markers > 1
        return Dict{String,Any}("status" => "failed",
            "failure_class" => "marker_conflict", "terminal" => true,
            "markers" => Dict("done" => done, "failed" => failed, "skipped" => skipped))
    elseif failed
        return Dict{String,Any}("status" => "failed",
            "failure_class" => "adapter_failure", "terminal" => true)
    elseif skipped
        return Dict{String,Any}("status" => "skipped",
            "failure_class" => "iteration_truncated", "terminal" => true)
    elseif done
        return Dict{String,Any}("status" => "completed",
            "failure_class" => nothing, "terminal" => true)
    end
    return Dict{String,Any}("status" => "pending",
        "failure_class" => "pending", "terminal" => false)
end

function _write_terminal_marker!(candidate_dir::String, status::String)
    status in ("completed", "failed", "skipped") || throw(ArgumentError("invalid terminal status"))
    mkpath(candidate_dir)
    marker = joinpath(candidate_dir, status == "completed" ? "done.ok" :
        status == "failed" ? "failed.ok" : "skipped.ok")
    isfile(marker) || touch(marker)
    return marker
end

function _materialize_terminal_artifact!(candidate_dir::String,
    status::AbstractDict)
    # The marker is the authoritative terminal source.  Reconcile the
    # canonical fields even when a candidate directory already contains a
    # stale status.json, while retaining unrelated diagnostic fields written
    # by the scorer.
    status_path = joinpath(candidate_dir, "status.json")
    materialized = Dict{String,Any}()
    if isfile(status_path)
        try
            existing = load_json(status_path)
            existing isa AbstractDict && merge!(materialized, existing)
        catch
            # A malformed stale artifact must not prevent terminal
            # reconciliation.  Replace it with the marker-derived record.
        end
    end
    merge!(materialized, Dict{String,Any}(String(k) => v for (k, v) in status))
    atomic_save_json(status_path, materialized; label="candidate_status")
    metrics_path = joinpath(candidate_dir, "metrics.json")
    if !isfile(metrics_path) && status["status"] != "pending"
        safe_save_json(metrics_path, Dict{String,Any}(
            "score" => Inf, "status" => status["status"],
            "failure_class" => get(status, "failure_class", nothing),
            "ranking_eligible" => false,
        ); label="candidate_metrics")
    end
    return status
end

"""Materialize the canonical terminal status and exactly one marker."""
function materialize_terminal_candidate!(candidate_dir::String, terminal_status::String;
    failure_class=nothing, details=Dict{String,Any}())
    terminal_status in ("completed", "failed", "skipped") ||
        throw(ArgumentError("invalid terminal status"))
    status = Dict{String,Any}(
        "status" => terminal_status,
        "terminal" => true,
        "failure_class" => failure_class,
    )
    for (key, value) in details
        status[String(key)] = value
    end
    # A rejected transition must never leave a stale success marker behind.
    for marker in ("done.ok", "failed.ok", "skipped.ok")
        path = joinpath(candidate_dir, marker)
        isfile(path) && rm(path)
    end
    _write_terminal_marker!(candidate_dir, terminal_status)
    _materialize_terminal_artifact!(candidate_dir, status)
    return status
end

"""Read terminal states without waiting or changing pending candidates."""
function normalized_iteration_result(candidate_dirs::AbstractVector{<:AbstractString};
    min_completion_fraction::Float64=0.9, iteration_truncated::Bool=false)
    done_count = 0
    failed_count = 0
    skipped_count = 0
    pending = String[]
    statuses = Any[]
    for candidate_dir in candidate_dirs
        status = candidate_terminal_status(String(candidate_dir))
        status["status"] != "pending" && _materialize_terminal_artifact!(String(candidate_dir), status)
        push!(statuses, merge(status, Dict("candidate_dir" => String(candidate_dir))))
        status["status"] == "completed" && (done_count += 1)
        status["status"] == "failed" && (failed_count += 1)
        status["status"] == "skipped" && (skipped_count += 1)
        status["status"] == "pending" && push!(pending, String(candidate_dir))
    end
    target_done = max(1, ceil(Int, length(candidate_dirs) *
        clamp(min_completion_fraction, 0.0, 1.0)))
    return Dict{String,Any}(
        "done" => done_count, "failed" => failed_count, "skipped" => skipped_count,
        "pending" => pending, "pending_count" => length(pending),
        "statuses" => statuses,
        "threshold_reached" => done_count >= target_done,
        # Skipped candidates are terminal, but they are evidence that the
        # population was not fully accounted for.  Keep this identical for
        # local and collected/Slurm-style collection.
        "iteration_truncated" => iteration_truncated || skipped_count > 0,
    )
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
        # Candidate directories and configs are recoverable work, not proof of
        # a committed iteration.  A committed iteration has all cross-file
        # state needed to resume without replaying CMA updates.
        manifest_path = joinpath(iter_dir, "iteration_commit.json")
        manifest_ok = false
        if isfile(manifest_path)
            try
                manifest = load_json(manifest_path)
                manifest_ok = get(manifest, "status", "") == "committed" &&
                    String(get(manifest, "stage", "")) == basename(stage_root) &&
                    Int(get(manifest, "iteration", -1)) == iter_idx
            catch
                manifest_ok = false
            end
        end
        committed = manifest_ok && isfile(top_candidates_file) &&
            isfile(joinpath(stage_root, "stage_state.json")) &&
            isfile(joinpath(stage_root, "full_reusable_state.json")) &&
            isfile(joinpath(stage_root, "iter_metrics.jsonl"))
        completed = committed
        isfile(cand) && push!(iter_infos, (iter_idx, cand, completed))
    end
    isempty(iter_infos) && return nothing
    function max_iter_info(items)
        best = items[1]
        for item in items[2:end]
            if item[1] > best[1]
                best = item
            end
        end
        return best
    end
    completed_iters = [info for info in iter_infos if info[3]]
    # Always inspect the newest iteration.  A newer partial iteration must
    # not be hidden by an older committed one during resume.
    chosen = max_iter_info(iter_infos)
    return Dict(
        "last_iter" => chosen[1],
        "last_iter_file" => chosen[2],
        "iteration_completed" => chosen[3],
        "found_iterations" => sort([info[1] for info in iter_infos]),
        "completed_iterations" => sort([info[1] for info in completed_iters]),
        "resume_iteration" => chosen[3] ? chosen[1] + 1 : chosen[1],
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
    seed_value = _expand_environment_variables(String(raw["seed_config"]), "seed_config")
    raw["seed_config"] = isabspath(seed_value) ? seed_value : normpath(joinpath(config_dir, seed_value))
    output_value = _expand_environment_variables(String(raw["output_dir"]), "output_dir")
    raw["output_dir"] = isabspath(output_value) ? output_value :
        normpath(joinpath(config_dir, output_value))
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
        float(get(raw["objective"], "min_completion_fraction", 0.9)),
        Int(get(raw["objective"], "finish_iter_delay", 30)),
        String(get(raw["objective"], "search_policy", "baseline")),
        float(get(raw["objective"], "temporal_jump_weight", 0.2)),
        float(get(raw["objective"], "infection_extrema_weight", 0.1)),
    )
    posterior_raw = get(raw, "posterior", Dict{String,Any}())
    posterior = PosteriorConfig(
        Bool(get(posterior_raw, "enabled", true)),
        String(get(posterior_raw, "likelihood", "diagonal_gaussian_weekly")),
        Int(get(posterior_raw, "draws", 500)),
        Int(get(posterior_raw, "warmup", 250)),
        Int(get(posterior_raw, "max_depth", 8)),
        float(get(posterior_raw, "step_size", 0.05)),
        float(get(posterior_raw, "temperature", 1.0)),
        float(get(posterior_raw, "transfer_covariance_inflation", 1.75)),
        float(get(posterior_raw, "transfer_sigma_multiplier", 1.25)),
        float(get(posterior_raw, "new_dimension_variance", 2.0)),
        float(get(posterior_raw, "immigrant_fraction", 0.20)),
    )
    gt_value = haskey(raw, "gt_dir") ?
        _expand_environment_variables(String(raw["gt_dir"]), "gt_dir") : nothing
    gt_dir = gt_value !== nothing ?
        (isabspath(gt_value) ? gt_value : normpath(joinpath(config_dir, gt_value))) :
        nothing
    external_sim = gt_dir !== nothing && haskey(raw, "julia_bin") && haskey(raw, "project_dir") && haskey(raw, "advanced_cli") ?
        ExternalSimConfig(
            gt_dir,
            _expand_environment_variables(String(raw["julia_bin"]), "julia_bin"),
            _expand_environment_variables(String(raw["project_dir"]), "project_dir"),
            _expand_environment_variables(String(raw["advanced_cli"]), "advanced_cli"),
            Bool(get(raw, "disable_compiled_modules", false)),
        ) :
        nothing
    stage_freeze = Dict{String,Vector{String}}()
    if haskey(raw, "stage_freeze")
        for (k, v) in raw["stage_freeze"]
            stage_freeze[String(k)] = [String(x) for x in v]
        end
    end
    initial_state = haskey(raw, "initial_state") ? Dict{String,Any}(String(k) => v for (k, v) in raw["initial_state"]) : nothing
    age_population_weights = haskey(raw, "age_population_weights") ?
        Dict(String(k) => float(v) for (k, v) in raw["age_population_weights"]) :
        copy(DEFAULT_AGE_POPULATION_WEIGHTS)
    total_age_weight = sum(values(age_population_weights))
    total_age_weight > 0.0 || error("age_population_weights must have a positive sum")
    age_population_weights = Dict(k => v / total_age_weight for (k, v) in age_population_weights)
    validation = haskey(raw, "validation") ?
        Dict(String(k) => v for (k, v) in raw["validation"]) :
        Dict{String,Any}("enabled" => true, "holdout_days" => 28, "seeds" => [42, 43, 44])
    temporal_parameterization = String(get(raw, "temporal_parameterization", "monthly"))
    temporal_parameterization in ("weekly", "monthly") ||
        error("Unsupported temporal_parameterization: $(temporal_parameterization)")
    runtime_seed = load_json(raw["seed_config"])
    _normalize_seed_paths!(runtime_seed, dirname(raw["seed_config"]))
    return OptimizerConfig(raw["seed_config"], raw["output_dir"], Int(raw["monthly_days"]), stages, scalar_bounds, temporal_bounds, scalar_preprocessing, temporal_parameterization, age_population_weights, validation, objective, external_sim, stage_freeze, initial_state, posterior, runtime_seed)
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

function stage_specs(seed::Dict{String,Any}, specs::Vector{ParamSpec}, cfg::OptimizerConfig, stage::StageConfig)
    freeze = get(cfg.stage_freeze, stage.name, String[])
    active_specs = ParamSpec[]
    for spec in specs
        spec.name in freeze && continue
        if spec.kind == :temporal
            active_length = temporal_active_length(seed, spec, stage.fit_months, cfg)
            push!(active_specs, ParamSpec(spec.name, spec.kind, max(active_length, 1), spec.lower, spec.upper))
        else
            push!(active_specs, spec)
        end
    end
    return active_specs
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
            if optcfg !== nothing && optcfg.temporal_parameterization == "monthly"
                interval_times = get_nested(seed, replace(spec.name, "interval_values" => "interval_times"))
                validate_interval_times(interval_times)
                for month in 1:spec.length
                    idxs = [i for (i, day) in enumerate(interval_times)
                            if monthly_bucket(day, optcfg.monthly_days) == month]
                    source_idx = isempty(idxs) ? (isempty(current) ? 0 : min(month, length(current))) : last(idxs)
                    push!(values, source_idx == 0 ? 0.5 : float(current[clamp(source_idx, 1, length(current))]))
                end
            else
                # A previous stage may contain fewer temporal buckets. Extend
                # it continuously instead of indexing beyond the shorter trajectory.
                available = min(spec.length, length(current))
                append!(values, map(float, current[1:available]))
                if available < spec.length
                    fill_value = available == 0 ? 0.5 : float(current[available])
                    append!(values, fill(fill_value, spec.length - available))
                end
            end
        end
    end
    return values
end

"""Inclusive 30-day buckets: day 1..30 is bucket 1, day 31..60 bucket 2."""
function monthly_bucket(day, monthly_days::Int)
    monthly_days > 0 || throw(ArgumentError("monthly_days must be positive"))
    isfinite(Float64(day)) && Float64(day) > 0 ||
        throw(ArgumentError("interval day must be positive and finite"))
    return fld(Int(ceil(Float64(day))) - 1, monthly_days) + 1
end

function validate_interval_times(interval_times)
    vals = Float64.(collect(interval_times))
    all(isfinite, vals) || throw(ArgumentError("interval_times must be finite"))
    all(>(0), vals) || throw(ArgumentError("interval_times must be positive"))
    all(diff(vals) .> 0) || throw(ArgumentError("interval_times must be strictly increasing"))
    return vals
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
    if cfg.temporal_parameterization == "monthly"
        return min(active_months, spec.length)
    end
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
    if cfg.temporal_parameterization == "monthly"
        return [
            ((month - 1) * cfg.monthly_days + 1, min(month * cfg.monthly_days, active_days))
            for month in 1:min(spec.length, active_months)
        ]
    end
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
    active_months >= 0 || throw(ArgumentError("active_months must be nonnegative"))
    monthly_days = optcfg === nothing ? 30 : optcfg.monthly_days
    set_nested!(cfg, "stop_simulation_time", active_months * monthly_days)
    # A zero-month conversion is an explicit empty-horizon policy: preserve
    # every seed coordinate and emit only the effective stop time.
    active_months == 0 && return cfg
    idx = 1
    for spec in specs
        if spec.kind == :scalar
            if idx <= length(x)
                val = optcfg === nothing ? x[idx] : decode_scalar_value(optcfg, seed, spec, x[idx])
                set_nested!(cfg, spec.name, val)
            end
            idx += 1
        else
            current = map(float, get_nested(cfg, spec.name))
            active = optcfg === nothing ? min(active_months, spec.length) : min(
                spec.length,
                temporal_active_length(seed, spec, active_months, optcfg),
            )
            if optcfg !== nothing && optcfg.temporal_parameterization == "monthly"
                interval_times = get_nested(seed, replace(spec.name, "interval_values" => "interval_times"))
                isempty(interval_times) || validate_interval_times(interval_times)
                active == 0 && (idx += spec.length; continue)
                for i in 1:length(current)
                    isempty(interval_times) && break
                    month = clamp(monthly_bucket(interval_times[min(i, length(interval_times))],
                        optcfg.monthly_days), 1, active)
                    # Entries beyond the requested horizon are inactive seed
                    # suffixes.  Do not map them to the last active bucket.
                    monthly_bucket(interval_times[min(i, length(interval_times))],
                        optcfg.monthly_days) > active && continue
                    idx_x = idx + month - 1
                    idx_x <= length(x) || continue
                    current[i] = x[idx_x]
                end
                # The vector layout still reserves the full specification
                # width, including inactive coordinates.
                idx += spec.length
            else
                for i in 1:active
                    idx <= length(x) && (current[i] = x[idx])
                    idx += 1
                end
                for _ in active+1:spec.length
                    idx += 1
                end
            end
            set_nested!(cfg, spec.name, current)
        end
    end
    return cfg
end

function inject_frozen!(cfg_out::Dict{String,Any}, seed::Dict{String,Any}, specs::Vector{ParamSpec}, frozen_names::Vector{String})
    for spec in specs
        spec.name in frozen_names || continue
        set_nested!(cfg_out, spec.name, get_nested(seed, spec.name))
    end
end

function inject_temporal_prefix_locks!(
    cfg_out::Dict{String,Any},
    seed::Dict{String,Any},
    previous_specs::Union{Nothing,Vector{ParamSpec}},
)
    previous_specs === nothing && return cfg_out
    for spec in previous_specs
        spec.kind == :temporal || continue
        old_values = get_nested(seed, spec.name)
        current_values = get_nested(cfg_out, spec.name)
        old_values isa AbstractVector || continue
        current_values isa AbstractVector || continue
        locked = copy(current_values)
        optcfg = CURRENT_OPTIMIZER_CONFIG[]
        if optcfg !== nothing && optcfg.temporal_parameterization == "monthly"
            interval_times = get_nested(seed, replace(spec.name, "interval_values" => "interval_times"))
            old_days = spec.length * optcfg.monthly_days
            for i in 1:min(length(interval_times), length(locked), length(old_values))
                float(interval_times[i]) <= old_days || continue
                locked[i] = old_values[i]
            end
        else
            n = min(spec.length, length(old_values), length(current_values))
            n == 0 && continue
            locked[1:n] .= old_values[1:n]
        end
        set_nested!(cfg_out, spec.name, locked)
    end
    return cfg_out
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

    cmd_args = String[
        simcfg.julia_bin,
        "--project=$(simcfg.project_dir)",
    ]
    simcfg.disable_compiled_modules && push!(cmd_args, "--compiled-modules=no")
    append!(cmd_args, [
        "--threads=4",
        simcfg.advanced_cli,
        config_path,
        "--output-daily",
        daily_path,
        "--output-summary",
        summary_path,
    ])
    cmd = Cmd(cmd_args)
    timeout_seconds = Float64(get(cfg.validation, "adapter_timeout_seconds", 3600.0))
    timeout_seconds > 0 || throw(ArgumentError("validation.adapter_timeout_seconds must be positive"))
    stdout_path = joinpath(workdir, "adapter.stdout.log")
    stderr_path = joinpath(workdir, "adapter.stderr.log")
    started = now(UTC)
    process = nothing
    timed_out = false
    success = false
    open(stdout_path, "w") do stdout_io
        open(stderr_path, "w") do stderr_io
            try
                process = run(pipeline(cmd, stdout=stdout_io, stderr=stderr_io); wait=false)
                wait_status = timedwait(() -> process_exited(process), timeout_seconds;
                                        pollint=min(0.25, timeout_seconds))
                timed_out = wait_status == :timed_out
                if timed_out
                    kill(process)
                    wait(process)
                end
                success = !timed_out && Base.success(process)
            catch err
                @warn "External simulation failed" err
            end
        end
    end
    invocation = Dict{String,Any}(
        "schema_version" => "adapter-invocation-v1", "command" => cmd_args,
        "working_directory" => workdir, "started_at" => string(started),
        "finished_at" => string(now(UTC)), "timeout_seconds" => timeout_seconds,
        "timed_out" => timed_out, "exit_code" => process === nothing ? nothing : process.exitcode,
        "success" => success, "stdout" => stdout_path, "stderr" => stderr_path)
    safe_save_json(joinpath(workdir, "adapter_invocation.json"), invocation;
                   label="adapter_invocation")
    return success, daily_path
end

function load_gt_series(gt_dir::String)
    function load_csv(name)
        path = joinpath(gt_dir, name)
        isfile(path) || return Union{Missing,Float64}[]
        parsed = _read_named_gt_csv(path)
        rows = Tuple{Int,Union{Missing,Float64}}[
            (day, value == -1 ? missing : value) for (day, value) in parsed]
        isempty(rows) && return Float64[]
        sort!(rows, by = first)
        max_day = rows[end][1]
        # Preserve the declared day identity.  Do not rebase a series whose
        # first observation starts after day one.
        values = Union{Missing,Float64}[missing for _ in 1:max_day]
        for (day, value) in rows
            values[day] = value
        end
        return values
    end
    function load_optional(name)
        path = joinpath(gt_dir, name)
        isfile(path) || return Float64[]
        # Optional files are already represented by the preflight manifest.
        # Runtime scoring must nevertheless be safe when a caller uses a
        # config loaded before that manifest was written, or when the file
        # changes between preflight and scoring.  An invalid optional source
        # is omitted rather than escaping a parser exception into ranking.
        try
            return load_csv(name)
        catch err
            @warn "Ignoring malformed optional ground-truth series" path err
            return Float64[]
        end
    end
    # Required files retain fail-closed parser behavior.  Missing files keep
    # the historical empty-series representation, which the scoring layer
    # converts to its deterministic Inf/no-data result.
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
    # Keep zero-observation metrics finite without letting one simulated
    # event produce an artificial million-scale ratio.
    denom = max(mean(abs.(b[1:n])), 1.0)
    return mae_series(a, b) / denom
end

"""
Return the finite, index-preserving pairs used by every score family.
Missing and non-finite values remove the same original day from both series;
neither input is independently compressed.  The returned day numbers are
one-based indices in the original trajectories.
"""
function paired_observations(gt, sim, days::Int)
    n = min(max(days, 0), length(gt), length(sim))
    out_gt = Float64[]
    out_sim = Float64[]
    indices = Int[]
    for day in 1:n
        gv = gt[day]
        sv = sim[day]
        if gv === missing || !(sv isa Real) || !isfinite(Float64(sv))
            continue
        end
        g = try Float64(gv) catch; continue end
        isfinite(g) || continue
        push!(out_gt, g)
        push!(out_sim, Float64(sv))
        push!(indices, day)
    end
    return out_gt, out_sim, indices
end

function score_payload_empty()
    return Dict{String,Any}(
        "periods" => Vector{Vector{Int}}(),
        "retained_indices" => Int[],
        "retained_count" => 0,
    )
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

function student_sparse_aligned_vectors(gt_daily::AbstractVector{T}, sim_daily::AbstractVector{Float64}, days::Int) where {T<:Union{Missing,Float64}}
    n = min(days, length(sim_daily), length(gt_daily))
    out_g = Float64[]
    out_s = Float64[]
    for day in 1:n
        gt_daily[day] === missing && continue
        push!(out_g, Float64(gt_daily[day]))
        push!(out_s, sim_daily[day])
    end
    return out_g, out_s
end

function per_trajectory_rmae(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Inf
    vals = Float64[]
    for traj in trajs
        gg, ss, _ = paired_observations(gt_series, traj, days)
        isempty(gg) || push!(vals, rmae_series(ss, gg))
    end
    isempty(vals) && return Inf
    return sum(vals) / length(vals)
end

function per_trajectory_cumulative_error(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Inf
    vals = Float64[]
    for traj in trajs
        gg, ss, _ = paired_observations(gt_series, traj, days)
        isempty(gg) && continue
        gc = cumulative_series(gg)
        sc = cumulative_series(ss)
        push!(vals, abs(last(sc) - last(gc)) / max(abs(last(gc)), 1.0))
    end
    isempty(vals) && return Inf
    return sum(vals) / length(vals)
end

function per_trajectory_blocked_cumulative_error(
    daily_path::String,
    metric::String,
    gt_series::AbstractVector{T} where T<:Union{Missing,Float64},
    days::Int;
    block_days::Int=28,
)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Inf
    n = min(days, length(gt_series))
    n <= 0 && return Inf
    block_errors = Float64[]
    for traj in trajs
        trajectory_errors = Float64[]
        for start_day in 1:block_days:n
            end_day = min(start_day + block_days - 1, n, length(traj))
            end_day < start_day && continue
            observed_values, simulated_values, valid = paired_observations(
                gt_series[start_day:end_day], traj[start_day:end_day], end_day - start_day + 1)
            isempty(valid) && continue
            observed = sum(observed_values)
            simulated = sum(simulated_values)
            push!(trajectory_errors, abs(simulated - observed) / max(abs(observed), 1.0))
        end
        isempty(trajectory_errors) || push!(block_errors, mean(trajectory_errors))
    end
    isempty(block_errors) && return Inf
    return mean(block_errors)
end

function validation_score_from_daily_legacy(cfg::OptimizerConfig, daily_path::String, days::Int)
    enabled = Bool(get(cfg.validation, "enabled", true))
    enabled || return Dict{String,Any}("enabled" => false)
    holdout_days = max(Int(get(cfg.validation, "holdout_days", 28)), 1)
    start_day = max(1, days - holdout_days + 1)
    gt = load_gt_series(cfg.external_sim === nothing ? joinpath(MANAGER_ROOT, "gt") : cfg.external_sim.gt_dir)
    metrics = Dict{String,Float64}()
    for metric in ("daily_detections", "daily_deaths", "daily_age_05_14_detections")
        haskey(gt, metric) || continue
        trajs = read_daily_metric(daily_path, metric)
        trajs === nothing && continue
        gt_slice = gt[metric][start_day:min(days, length(gt[metric]))]
        valid_gt = [x for x in gt_slice if x !== missing]
        isempty(valid_gt) && continue
        per_trajectory = Float64[]
        for traj in trajs
            end_day = min(days, length(traj), length(gt[metric]))
            end_day < start_day && continue
            sim_slice = Float64.(traj[start_day:end_day])
            gt_values = Float64[
                Float64(gt[metric][day])
                for day in start_day:end_day
                if gt[metric][day] !== missing
            ]
            length(sim_slice) == length(gt_values) || continue
            push!(per_trajectory, rmae_series(sim_slice, gt_values))
        end
        isempty(per_trajectory) || (metrics[metric] = mean(per_trajectory))
    end
    values = collect(values(metrics))
    return Dict{String,Any}(
        "enabled" => true,
        "start_day" => start_day,
        "end_day" => days,
        "holdout_days" => holdout_days,
        "seeds" => get(cfg.validation, "seeds", [42, 43, 44]),
        "metrics" => metrics,
        "mean_error" => isempty(values) ? Inf : mean(values),
    )
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
    vals = Float64[]
    for traj in trajs
        g, s, _ = paired_observations(gt_series, traj, days)
        isempty(g) && continue
        push!(vals, rmae_series(rolling_sum(s, 7), rolling_sum(g, 7)))
    end
    return vals
end

function weekly_error_distributions_legacy(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
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
    if metric == "daily_student_detections"
        observations = Float64[]
        predictions = Float64[]
        absolute_errors = Float64[]
        normalized_absolute_errors = Float64[]
        maes = Float64[]
        rmaes = Float64[]
        periods = Vector{Vector{Int}}()
        n = min(days, length(gt_series))
        for day in 1:n
            gt_series[day] === missing && continue
            sim_values = [Float64(traj[day]) for traj in trajs if day <= length(traj)]
            isempty(sim_values) && continue
            observation = Float64(gt_series[day])
            prediction = mean(sim_values)
            residual = prediction - observation
            scale = max(abs(observation), 1.0)
            push!(absolute_errors, abs(residual))
            push!(normalized_absolute_errors, abs(residual) / scale)
            push!(maes, mean(abs.(sim_values .- observation)))
            push!(rmaes, mean(abs.(sim_values .- observation)) / scale)
            push!(periods, [day, day])
            push!(observations, observation)
            push!(predictions, prediction)
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
            weekly_error = abs(sum(sim_week) - sum(paired_gt))
            weekly_scale = max(abs(sum(paired_gt)), 1.0)
            absolute_error += weekly_error
            normalized_absolute_error += weekly_error / weekly_scale
            mae += weekly_error
            rmae += weekly_error / weekly_scale
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

"""
Index-preserving weekly implementation.  This later method intentionally
supersedes the historical implementation above so every caller, including
collection/Slurm reconstruction, shares one paired-day policy.
"""
function weekly_error_distributions(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    empty_payload = Dict{String,Any}(
        "absolute_error" => Float64[], "normalized_absolute_error" => Float64[],
        "mae" => Float64[], "rmae" => Float64[],
        "periods" => Vector{Vector{Int}}(), "retained_indices" => Vector{Vector{Int}}(),
        "observations" => Float64[], "predictions" => Float64[])
    trajs === nothing && return empty_payload
    n = min(max(days, 0), length(gt_series))
    result = deepcopy(empty_payload)
    for start_day in 1:7:n
        end_day = min(start_day + 6, n)
        per_traj = Tuple{Float64,Float64,Float64,Vector{Int}}[]
        for traj in trajs
            start_day > length(traj) && continue
            g, s, idx = paired_observations(gt_series[start_day:end_day],
                                             traj[start_day:min(end_day, length(traj))],
                                             end_day - start_day + 1)
            isempty(idx) && continue
            original = [start_day + i - 1 for i in idx]
            if metric == "daily_student_detections"
                observed = sum(g) / 7.0
                predicted = sum(s) / length(s)
            else
                observed = sum(g)
                predicted = sum(s)
            end
            err = abs(predicted - observed)
            push!(per_traj, (err, err / max(abs(observed), 1.0), observed, original))
        end
        isempty(per_traj) && continue
        push!(result["absolute_error"], mean(first.(per_traj)))
        push!(result["normalized_absolute_error"], mean(getindex.(per_traj, 2)))
        push!(result["mae"], mean(first.(per_traj)))
        push!(result["rmae"], mean(getindex.(per_traj, 2)))
        push!(result["periods"], [start_day, end_day])
        retained = unique(sort(vcat([x[4] for x in per_traj]...)))
        push!(result["retained_indices"], retained)
        push!(result["observations"], mean(getindex.(per_traj, 3)))
        push!(result["predictions"], mean([
            metric == "daily_student_detections" ?
                sum(paired_observations(gt_series[start_day:end_day],
                    (start_day > length(traj) ? Float64[] : traj[start_day:min(end_day, length(traj))]),
                    end_day - start_day + 1)[2]) /
                length(paired_observations(gt_series[start_day:end_day],
                    (start_day > length(traj) ? Float64[] : traj[start_day:min(end_day, length(traj))]),
                    end_day - start_day + 1)[2]) :
                sum(paired_observations(gt_series[start_day:end_day],
                    (start_day > length(traj) ? Float64[] : traj[start_day:min(end_day, length(traj))]),
                    end_day - start_day + 1)[2])
            for traj in trajs if !isempty(paired_observations(
                gt_series[start_day:end_day],
                (start_day > length(traj) ? Float64[] : traj[start_day:min(end_day, length(traj))]),
                end_day - start_day + 1)[3])
        ]))
    end
    return result
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

const TEMPORAL_ERROR_LOOKAHEAD = 5
const TEMPORAL_ERROR_WEIGHTS = fill(0.2, TEMPORAL_ERROR_LOOKAHEAD)

function age_population_share(cfg::OptimizerConfig, metric::String)
    startswith(metric, "daily_age_") || return 1.0
    occursin("daily_age_total_", metric) && return 1.0
    match_result = match(r"daily_age_(00_04|05_14|15_34|35_59|60_79|80_plus)_(detections|deaths)$", metric)
    match_result === nothing && return 1.0
    group = match_result.captures[1]
    return get(cfg.age_population_weights, group, 0.0)
end

function forward_error_average(values::AbstractVector{Float64}, start_index::Int)
    isempty(values) && return 0.0
    first_index = clamp(start_index, 1, length(values))
    last_index = min(length(values), first_index + TEMPORAL_ERROR_LOOKAHEAD - 1)
    weights = TEMPORAL_ERROR_WEIGHTS[1:(last_index - first_index + 1)]
    return sum(values[first_index:last_index] .* weights) / sum(weights)
end

function weekly_control_score(cfg::OptimizerConfig, daily_path::String, gt::AbstractDict, days::Int)
    metric_scores = Float64[]
    metric_weights = Float64[]
    available_groups = Dict(
        "detections" => any(haskey(gt, "daily_age_$(group)_detections") for group in keys(cfg.age_population_weights)),
        "deaths" => any(haskey(gt, "daily_age_$(group)_deaths") for group in keys(cfg.age_population_weights)),
    )
    for metric in WEEKLY_CONTROL_METRICS
        haskey(gt, metric) || continue
        if occursin("daily_age_total_", metric)
            suffix = endswith(metric, "_detections") ? "detections" : "deaths"
            available_groups[suffix] && continue
        end
        errors = weekly_error_distributions(daily_path, metric, gt[metric], days)
        rmaes = errors["rmae"]
        normalized_absolute_errors = errors["normalized_absolute_error"]
        isempty(rmaes) && continue
        n = min(length(rmaes), length(normalized_absolute_errors))
        score = mean(0.7 .* rmaes[1:n] .+ 0.3 .* normalized_absolute_errors[1:n])
        weight = age_population_share(cfg, metric)
        if isfinite(score) && weight > 0.0
            push!(metric_scores, score)
            push!(metric_weights, weight)
        end
    end
    isempty(metric_scores) && return Inf
    return sum(metric_scores .* metric_weights) / sum(metric_weights)
end

function temporal_jump_penalty(cfg::OptimizerConfig, candidate::AbstractDict)
    # Penalize the raw configured buckets, before MocosSimLauncher applies
    # any runtime seasonal multiplier to infection transmission.
    penalties = Float64[]
    for path in keys(cfg.temporal_bounds)
        values = try
            Float64.(get_nested(candidate, path))
        catch
            continue
        end
        length(values) < 2 && continue
        first_difference = mean(abs.(diff(values)))
        second_difference = length(values) < 3 ? 0.0 : mean(abs.(diff(values, 2)))
        push!(penalties, first_difference + second_difference)
    end
    return isempty(penalties) ? 0.0 : mean(penalties)
end

function infection_extrema_penalty(cfg::OptimizerConfig, candidate::AbstractDict)
    path = "infection_modulation.params.interval_values"
    values = try
        Float64.(get_nested(candidate, path))
    catch
        return 0.0
    end
    length(values) < 3 && return 0.0
    differences = diff(values)
    signs = [sign(value) for value in differences if abs(value) > 1e-12]
    length(signs) < 2 && return 0.0
    direction_changes = count(i -> signs[i] != signs[i - 1], 2:length(signs))
    boundary_hits = count(value -> value <= 1e-9 || value >= 1.0 - 1e-9, values)
    return direction_changes / (length(signs) - 1) + boundary_hits / length(values)
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
        if occursin("daily_age_total_", metric)
            suffix = endswith(metric, "_detections") ? "detections" : "deaths"
            has_groups = any(
                haskey(gt, "daily_age_$(group)_$(suffix)")
                for group in keys(cfg.age_population_weights)
            )
            has_groups && continue
        end
        errors_by_metric[metric] = weekly_error_distributions(daily_path, metric, gt[metric], days)
    end
    errors = zeros(Float64, spec.length)
    for (bi, (start_day, end_day)) in enumerate(bucket_ranges)
        bucket_values = Float64[]
        bucket_weights = Float64[]
        for (metric, metric_errors) in errors_by_metric
            periods = metric_errors["periods"]
            rmaes = metric_errors["rmae"]
            normalized_absolute_errors = metric_errors["normalized_absolute_error"]
            isempty(periods) && continue
            first_period = findfirst(period -> period[1] >= start_day, periods)
            first_period === nothing && continue
            n = min(length(rmaes), length(normalized_absolute_errors))
            first_period > n && continue
            period_scores = Float64[
                0.7 * rmaes[i] + 0.3 * normalized_absolute_errors[i]
                for i in 1:n
            ]
            push!(bucket_values, forward_error_average(period_scores, first_period))
            push!(bucket_weights, age_population_share(cfg, metric))
        end
        errors[bi] = isempty(bucket_values) ? 0.0 :
            sum(bucket_values .* bucket_weights) / sum(bucket_weights)
    end
    return errors
end

function cumulative_metric_values_legacy(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    g = drop_missing(gt_series[1:min(end, days)])
    g = cumulative_series(g)
    vals = Float64[]
    for traj in trajs
        s = Float64.(traj[1:min(end, days)])
        s = cumulative_series(s)
        push!(vals, abs(last(s) - last(g)) / max(abs(last(g)), 1.0))
    end
    return vals
end

function cumulative_error_distribution(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    vals = Float64[]
    for traj in trajs
        g, s, _ = paired_observations(gt_series, traj, days)
        isempty(g) && continue
        gc = cumulative_series(g)
        sc = cumulative_series(s)
        push!(vals, abs(last(sc) - last(gc)) / max(abs(last(gc)), 1.0))
    end
    return vals
end

function cumulative_metric_values(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    trajs = read_daily_metric(daily_path, metric)
    trajs === nothing && return Float64[]
    vals = Float64[]
    for traj in trajs
        g, s, _ = paired_observations(gt_series, traj, days)
        isempty(g) && continue
        gc, sc = cumulative_series(g), cumulative_series(s)
        push!(vals, abs(last(sc) - last(gc)) / max(abs(last(gc)), 1.0))
    end
    return vals
end

function cumulative_error_distribution_legacy(daily_path::String, metric::String, gt_series::AbstractVector{T} where T<:Union{Missing,Float64}, days::Int)
    return cumulative_metric_values_legacy(daily_path, metric, gt_series, days)
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
    return Float64[forward_error_average(guidance, bi) for bi in eachindex(guidance)]
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
    return Float64[forward_error_average(errors, bi) for bi in eachindex(errors)]
end

function vector_likelihood_payload(
    daily_path::String,
    gt::AbstractDict,
    days::Int;
    family::String="diagonal_gaussian_weekly",
    metric_names::Union{Nothing,AbstractVector}=nothing,
    dispersions::AbstractDict=Dict{String,Float64}(),
)
    family in ("diagonal_gaussian_weekly", "negative_binomial_weekly") ||
        error("Unsupported vector likelihood family: $family")
    dimensions = Any[]
    log_likelihood = 0.0
    selected_metrics = metric_names === nothing ? String.(collect(keys(gt))) :
        unique(String.(metric_names))
    missing_metrics = [metric for metric in selected_metrics if !haskey(gt, metric)]
    isempty(missing_metrics) || throw(ArgumentError(
        "likelihood metrics missing from ground truth: $(join(missing_metrics, ", "))"))
    isempty(selected_metrics) && throw(ArgumentError(
        "likelihood metric selection must not be empty"))
    for metric in sort!(unique(selected_metrics))
        errors = weekly_error_distributions(daily_path, metric, gt[metric], days)
        periods = errors["periods"]
        observations = errors["observations"]
        predictions = errors["predictions"]
        length(periods) == length(observations) == length(predictions) || continue
        for i in eachindex(periods)
            observation = observations[i]
            prediction = predictions[i]
            residual = prediction - observation
            scale = max(abs(observation), 1.0)
            standardized_residual = residual / scale
            default_dispersion = metric == "daily_deaths" ? 10.0 :
                                 metric == "daily_hospitalizations" ? 15.0 : 25.0
            dispersion = Float64(get(dispersions, metric, default_dispersion))
            isfinite(dispersion) && dispersion > 0 || throw(ArgumentError(
                "likelihood dispersion for $metric must be positive and finite"))
            contribution = if family == "negative_binomial_weekly"
                negative_binomial_loglikelihood(
                    round(Int, max(observation, 0.0)), max(prediction, 0.0), dispersion)
            else
                -0.5 * (standardized_residual^2 + log(2.0 * pi * scale^2))
            end
            log_likelihood += contribution
            push!(dimensions, Dict(
                "metric" => metric,
                "period" => periods[i],
                "observation" => observation,
                "prediction" => prediction,
                "residual" => residual,
                "scale" => scale,
                "dispersion" => family == "negative_binomial_weekly" ? dispersion : nothing,
                "standardized_residual" => standardized_residual,
                "log_likelihood_contribution" => contribution,
            ))
        end
    end
    return Dict(
        "family" => family,
        "scale_policy" => family == "negative_binomial_weekly" ?
            "metric-specific fixed dispersion" : "max(abs(observation), 1.0)",
        "log_likelihood" => log_likelihood,
        "dimensions" => dimensions,
    )
end

function likelihood_metric_names(cfg::OptimizerConfig)
    configured = get(cfg.validation, "likelihood_metrics", [
        "daily_detections", "daily_deaths", "daily_hospitalizations"])
    configured isa AbstractVector || throw(ArgumentError(
        "validation.likelihood_metrics must be an array"))
    names = String.(configured)
    isempty(names) && throw(ArgumentError(
        "validation.likelihood_metrics must not be empty"))
    return names
end

function likelihood_dispersions(cfg::OptimizerConfig)
    configured = get(cfg.validation, "likelihood_dispersions", Dict{String,Any}())
    configured isa AbstractDict || throw(ArgumentError(
        "validation.likelihood_dispersions must be an object"))
    result = Dict{String,Float64}()
    for (metric, value) in configured
        dispersion = Float64(value)
        isfinite(dispersion) && dispersion > 0 || throw(ArgumentError(
            "validation.likelihood_dispersions.$metric must be positive and finite"))
        result[String(metric)] = dispersion
    end
    return result
end

const OBJECTIVE_METRIC_DEFAULTS = Dict(
    "daily_detections" => 1.0,
    "daily_hospitalizations" => 0.0,
    "daily_deaths" => 1.0,
    "daily_student_detections" => 0.0,
    "daily_detections_cumulative" => 1.0,
    "daily_detections_cumulative_blocked" => 0.5,
    "daily_hospitalizations_cumulative" => 0.0,
    "daily_deaths_cumulative" => 1.0,
    "daily_deaths_cumulative_blocked" => 0.5,
    "daily_student_detections_cumulative" => 0.0,
)

function objective_score_legacy(
    cfg::OptimizerConfig,
    metrics::AbstractDict,
    weekly_control::Float64,
    jump_penalty::Float64,
    extrema_penalty::Float64,
)
    # Keep the compatibility entry point on the canonical guarded path so
    # local and collected scoring cannot diverge on non-finite optional terms.
    return objective_score(cfg, metrics, weekly_control, jump_penalty, extrema_penalty)
end

function score_with_real_sim(cfg::OptimizerConfig, candidate::Dict{String,Any}, days::Int; workdir::String)
    cfg.external_sim === nothing && throw(ArgumentError("External simulation config not provided"))
    # Parse required GT before launching the adapter.  This keeps malformed
    # required inputs fail-closed and prevents a candidate run from masking a
    # deterministic configuration/data error.
    gt = load_gt_series(cfg.external_sim.gt_dir)
    sim_ok, daily_path = run_external_sim(cfg, candidate, days; workdir=workdir)
    sim_ok || return Inf, Dict("sim_failed" => true)
    windows = stage_data_split(days, cfg.validation)
    training_days = Int(windows["train"]["end_day"])
    weekly_control = weekly_control_score(cfg, daily_path, gt, training_days)
    vector_likelihood = vector_likelihood_payload(daily_path, gt, training_days;
        family=cfg.posterior.likelihood, metric_names=likelihood_metric_names(cfg),
        dispersions=likelihood_dispersions(cfg))
    # Validation carries structured diagnostics (window, retained indices,
    # and per-metric scores), so keep the score manifest heterogeneous.
    metrics = Dict{String,Any}()
    for (metric, gtvals) in gt
        isempty(gtvals) && continue
        metrics[metric] = per_trajectory_rmae(daily_path, metric, drop_missing(gtvals), training_days)
        metrics["$(metric)_cumulative"] = per_trajectory_cumulative_error(daily_path, metric, drop_missing(gtvals), training_days)
        metrics["$(metric)_cumulative_blocked"] = per_trajectory_blocked_cumulative_error(daily_path, metric, gtvals, training_days)
    end
    metrics["weekly_control_score"] = weekly_control
    jump_penalty = temporal_jump_penalty(cfg, candidate)
    metrics["temporal_jump_penalty"] = jump_penalty
    extrema_penalty = infection_extrema_penalty(cfg, candidate)
    metrics["infection_extrema_penalty"] = extrema_penalty
    metrics["vector_log_likelihood"] = vector_likelihood["log_likelihood"]
    cfg.posterior.likelihood == "negative_binomial_weekly" &&
        (metrics["negative_binomial_log_likelihood"] =
            -Float64(vector_likelihood["log_likelihood"]))
    validation = validation_score_from_daily(cfg, daily_path, days)
    metrics["validation_mean_error"] = Float64(get(validation, "mean_error", Inf))
    metrics["validation_window"] = validation
    return objective_score(cfg, metrics, weekly_control, jump_penalty, extrema_penalty), metrics
end

function score_from_daily(cfg::OptimizerConfig, daily_path::String, days::Int, candidate::Union{Nothing,AbstractDict}=nothing)
    gt = load_gt_series(cfg.external_sim.gt_dir)
    windows = stage_data_split(days, cfg.validation)
    training_days = Int(windows["train"]["end_day"])
    weekly_control = weekly_control_score(cfg, daily_path, gt, training_days)
    vector_likelihood = vector_likelihood_payload(daily_path, gt, training_days;
        family=cfg.posterior.likelihood, metric_names=likelihood_metric_names(cfg),
        dispersions=likelihood_dispersions(cfg))
    # The validation window is a structured diagnostic, not a scalar metric.
    # A heterogeneous payload prevents assigning it to a Float64-only dict.
    metrics = Dict{String,Any}()
    for (metric, gtvals) in gt
        isempty(gtvals) && continue
        metrics[metric] = per_trajectory_rmae(daily_path, metric, gtvals, training_days)
        metrics["$(metric)_cumulative"] = per_trajectory_cumulative_error(daily_path, metric, gtvals, training_days)
        metrics["$(metric)_cumulative_blocked"] = per_trajectory_blocked_cumulative_error(daily_path, metric, gtvals, training_days)
    end
    metrics["weekly_control_score"] = weekly_control
    jump_penalty = candidate === nothing ? 0.0 : temporal_jump_penalty(cfg, candidate)
    metrics["temporal_jump_penalty"] = jump_penalty
    extrema_penalty = candidate === nothing ? 0.0 : infection_extrema_penalty(cfg, candidate)
    metrics["infection_extrema_penalty"] = extrema_penalty
    metrics["vector_log_likelihood"] = vector_likelihood["log_likelihood"]
    cfg.posterior.likelihood == "negative_binomial_weekly" &&
        (metrics["negative_binomial_log_likelihood"] =
            -Float64(vector_likelihood["log_likelihood"]))
    validation = validation_score_from_daily(cfg, daily_path, days)
    metrics["validation_mean_error"] = Float64(get(validation, "mean_error", Inf))
    metrics["validation_window"] = validation
    return objective_score(cfg, metrics, weekly_control, jump_penalty, extrema_penalty), metrics
end

"""Describe the effective weighted objective terms before arithmetic.

The manifest intentionally remains heterogeneous: scoring diagnostics can
contain structured validation payloads alongside scalar metrics.  Optional
terms are enabled only when their effective weight is positive and their
source is present; a non-finite value is then a deterministic failure rather
than an accidental `0 * Inf` NaN.
"""
function effective_metric_manifest(
    cfg::OptimizerConfig,
    metrics::AbstractDict,
    weekly_control::Real,
    jump_penalty::Real,
    extrema_penalty::Real,
)
    weights = cfg.objective.weights
    manifest = Dict{String,Any}()
    for (metric, default_weight) in OBJECTIVE_METRIC_DEFAULTS
        weight = Float64(get(weights, metric, default_weight))
        present = haskey(metrics, metric)
        manifest[metric] = Dict(
            "enabled" => present && weight > 0.0,
            "required" => present && weight > 0.0,
            "weight" => weight,
            "source_present" => present,
            "all_missing_policy" => "Inf",
        )
    end
    for metric in keys(weights)
        metric in keys(OBJECTIVE_METRIC_DEFAULTS) && continue
        metric in ("weekly_control", "temporal_jump_penalty",
                   "infection_extrema_penalty") && continue
        haskey(manifest, metric) && continue
        manifest[metric] = Dict(
            "enabled" => haskey(metrics, metric) && weights[metric] > 0.0,
            "required" => haskey(metrics, metric) && weights[metric] > 0.0,
            "weight" => Float64(weights[metric]),
            "source_present" => haskey(metrics, metric),
            "all_missing_policy" => "Inf",
        )
    end
    weekly_weight = Float64(get(weights, "weekly_control", 1.0))
    manifest["weekly_control"] = Dict(
        "enabled" => weekly_weight > 0.0,
        "required" => weekly_weight > 0.0,
        "weight" => weekly_weight,
        "source_present" => true,
        "finite" => isfinite(weekly_control),
        "all_missing_policy" => "Inf",
    )
    jump_weight = cfg.objective.temporal_jump_weight
    manifest["temporal_jump_penalty"] = Dict(
        "enabled" => jump_weight > 0.0,
        "required" => jump_weight > 0.0,
        "weight" => jump_weight,
        "source_present" => true,
        "finite" => isfinite(jump_penalty),
        "all_missing_policy" => "Inf",
    )
    extrema_weight = cfg.objective.infection_extrema_weight
    manifest["infection_extrema_penalty"] = Dict(
        "enabled" => extrema_weight > 0.0,
        "required" => extrema_weight > 0.0,
        "weight" => extrema_weight,
        "source_present" => true,
        "finite" => isfinite(extrema_penalty),
        "all_missing_policy" => "Inf",
    )
    return manifest
end

function objective_score(
    cfg::OptimizerConfig,
    metrics::AbstractDict,
    weekly_control::Float64,
    jump_penalty::Float64,
    extrema_penalty::Float64,
)
    weights = cfg.objective.weights
    total = 0.0
    manifest = effective_metric_manifest(
        cfg, metrics, weekly_control, jump_penalty, extrema_penalty)
    invalid = false
    for (metric, default_weight) in OBJECTIVE_METRIC_DEFAULTS
        weight = Float64(get(weights, metric, default_weight))
        value_present = haskey(metrics, metric)
        value = value_present ? try Float64(metrics[metric]) catch; Inf end : Inf
        value_present || continue
        weight > 0.0 && !isfinite(value) && (invalid = true)
        weight <= 0.0 && continue
        total += weight * value
    end
    wc_weight = Float64(get(weights, "weekly_control", 1.0))
    wc_weight > 0.0 && !isfinite(weekly_control) && (invalid = true)
    wc_weight > 0.0 && (total += wc_weight * weekly_control)
    for (metric, raw_value) in metrics
        metric in keys(OBJECTIVE_METRIC_DEFAULTS) && continue
        metric in ("weekly_control_score", "temporal_jump_penalty",
                   "infection_extrema_penalty", "vector_log_likelihood",
                   "validation_mean_error", "validation_window") && continue
        haskey(weights, metric) || continue
        weight = Float64(weights[metric])
        value = try Float64(raw_value) catch; Inf end
        weight > 0.0 && !isfinite(value) && (invalid = true)
        weight <= 0.0 && continue
        total += weight * value
    end
    jump_weight = cfg.objective.temporal_jump_weight
    extrema_weight = cfg.objective.infection_extrema_weight
    jump_weight > 0.0 && !isfinite(jump_penalty) && (invalid = true)
    extrema_weight > 0.0 && !isfinite(extrema_penalty) && (invalid = true)
    jump_weight > 0.0 && (total += jump_weight * jump_penalty)
    extrema_weight > 0.0 && (total += extrema_weight * extrema_penalty)
    metrics isa Dict{String,Any} &&
        (metrics["effective_metric_manifest"] = manifest)
    !invalid && isfinite(total) ? total : Inf
end

function validation_score_from_daily(cfg::OptimizerConfig, daily_path::String, days::Int)
    enabled = Bool(get(cfg.validation, "enabled", true))
    enabled || return Dict{String,Any}("enabled" => false, "mean_error" => 0.0,
                                       "retained_indices" => Int[], "count" => 0)
    windows = stage_data_split(days, cfg.validation)
    requested_start = Int(windows["validation"]["start_day"])
    requested_end = Int(windows["validation"]["end_day"])
    holdout_days = requested_end - requested_start + 1
    gt = load_gt_series(cfg.external_sim === nothing ? joinpath(MANAGER_ROOT, "gt") : cfg.external_sim.gt_dir)
    metric_scores = Dict{String,Float64}()
    retained = Int[]
    for metric in ("daily_detections", "daily_deaths", "daily_age_05_14_detections")
        haskey(gt, metric) || continue
        trajs = read_daily_metric(daily_path, metric)
        trajs === nothing && continue
        scores = Float64[]
        for traj in trajs
            requested_start > min(requested_end, length(gt[metric]), length(traj)) && continue
            g, s, idx = paired_observations(
                gt[metric][requested_start:min(requested_end, length(gt[metric]))],
                traj[requested_start:min(requested_end, length(traj))],
                min(requested_end, length(gt[metric]), length(traj)) - requested_start + 1)
            isempty(idx) && continue
            push!(scores, rmae_series(s, g))
            append!(retained, requested_start .+ idx .- 1)
        end
        isempty(scores) || (metric_scores[metric] = mean(scores))
    end
    retained = unique(sort(retained))
    actual_start = isempty(retained) ? requested_start : first(retained)
    actual_end = isempty(retained) ? 0 : last(retained)
    return Dict{String,Any}(
        "enabled" => true, "start_day" => actual_start, "end_day" => actual_end,
        "requested_start_day" => requested_start, "holdout_days" => holdout_days,
        "split_mode" => windows["mode"],
        "seeds" => get(cfg.validation, "seeds", [42, 43, 44]),
        "metrics" => metric_scores, "retained_indices" => retained,
        "count" => length(retained),
        "mean_error" => isempty(metric_scores) ? Inf : mean(collect(values(metric_scores))),
    )
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

function wait_for_iteration_outputs(list_file::String; poll::Float64=10.0,
    min_completion_fraction::Float64=0.9, finish_iter_delay::Int=30,
    max_wait::Float64=Inf)
    cand_dirs = [String(strip(x)) for x in readlines(list_file) if !isempty(strip(x))]
    target_done = max(1, ceil(Int, length(cand_dirs) * clamp(min_completion_fraction, 0.0, 1.0)))
    threshold_reached_at = nothing
    started_at = time()
    while true
        done_count = 0
        failed_count = 0
        skipped_count = 0
        pending = String[]
        statuses = Any[]
        for d in cand_dirs
            status = candidate_terminal_status(d)
            status["status"] != "pending" && _materialize_terminal_artifact!(d, status)
            push!(statuses, merge(status, Dict("candidate_dir" => d)))
            if status["status"] == "completed"
                done_count += 1
            elseif status["status"] == "failed"
                failed_count += 1
            elseif status["status"] == "skipped"
                skipped_count += 1
            else
                push!(pending, d)
            end
        end

        if isempty(pending)
            return Dict(
                "done" => done_count,
                "failed" => failed_count,
                "skipped" => skipped_count, "pending" => String[],
                "pending_count" => 0, "statuses" => statuses,
                "threshold_reached" => done_count >= target_done,
                "iteration_truncated" => skipped_count > 0,
            )
        end

        if done_count >= target_done
            if threshold_reached_at === nothing
                threshold_reached_at = time()
            end
            if (time() - threshold_reached_at) >= finish_iter_delay
                for d in pending
                    _write_terminal_marker!(d, "skipped")
                end
                return wait_for_iteration_outputs(list_file; poll=poll,
                    min_completion_fraction=min_completion_fraction,
                    finish_iter_delay=finish_iter_delay, max_wait=0.0)
            end
        end

        if (time() - started_at) >= max_wait
            for d in pending
                _write_terminal_marker!(d, "skipped")
            end
            final = wait_for_iteration_outputs(list_file; poll=poll,
                min_completion_fraction=min_completion_fraction,
                finish_iter_delay=finish_iter_delay, max_wait=0.0)
            final["iteration_truncated"] = true
            return final
        end

        if threshold_reached_at !== nothing && finish_iter_delay <= 0
            for d in pending
                _write_terminal_marker!(d, "skipped")
            end
            final = wait_for_iteration_outputs(list_file; poll=poll,
                min_completion_fraction=min_completion_fraction,
                finish_iter_delay=finish_iter_delay, max_wait=0.0)
            final["iteration_truncated"] = true
            return final
        end

        sleep(max(poll, 0.0))
    end
end

function submit_slurm_array(cfg::OptimizerConfig, list_file::String)
    simcfg = cfg.external_sim
    lines = readlines(list_file)
    n = length(lines)
    n == 0 && return ""
    timeout_seconds = Float64(get(cfg.validation, "adapter_timeout_seconds", 3600.0))
    cmd = `sbatch --parsable -c 4 -t 01:15:00 --mem=20G --array=0-$(n-1) scripts/score_candidates.sh $list_file $(simcfg.julia_bin) $(simcfg.project_dir) $(simcfg.advanced_cli) $(simcfg.gt_dir) $timeout_seconds`
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
        "schema_version" => "experiment-v1",
        "experiment_type" => "cma_candidate",
        "score" => combined,
        "metrics" => metrics,
        "early_reject" => false,
        "simulated" => "real",
        "status" => isfinite(combined) ? "completed" : "failed",
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
            result["vector_likelihood"] = vector_likelihood_payload(
                daily_path, load_gt_series(cfg.external_sim.gt_dir), days;
                family=cfg.posterior.likelihood,
                metric_names=likelihood_metric_names(cfg),
                dispersions=likelihood_dispersions(cfg),
            )
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

function run_validation_replicates(
    cfg::OptimizerConfig,
    candidate::Dict{String,Any},
    days::Int;
    workdir::String=joinpath(cfg.output_dir, "validation_replicates"),
)
    enabled = Bool(get(cfg.validation, "enabled", true))
    enabled || return Dict{String,Any}("enabled" => false)
    seeds = Int.(get(cfg.validation, "seeds", [42, 43, 44]))
    results = Any[]
    for seed in seeds
        replicate = deepcopy(candidate)
        replicate["params_seed"] = seed
        replicate_dir = joinpath(workdir, "seed_$(seed)")
        result = score_candidate(replicate, cfg, days; workdir=replicate_dir)
        push!(results, Dict(
            "seed" => seed,
            "score" => result["score"],
            "status" => result["status"],
            "metrics" => result["metrics"],
        ))
    end
    finite_scores = Float64[
        Float64(result["score"])
        for result in results
        if get(result, "status", "") == "completed" && isfinite(Float64(result["score"]))
    ]
    return Dict{String,Any}(
        "enabled" => true,
        "seeds" => seeds,
        "results" => results,
        "mean_score" => isempty(finite_scores) ? Inf : mean(finite_scores),
        "std_score" => length(finite_scores) < 2 ? 0.0 : std(finite_scores),
    )
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

const MODULATION_LIPSCHITZ_DELTA = 0.15

function clip_candidate(candidate::Vector{Float64}, specs_stage::Vector{ParamSpec})
    evaluated = copy(candidate)
    lower_hits = Bool[]
    upper_hits = Bool[]
    lipschitz_hits = Bool[]
    idx = 1
    for spec in specs_stage
        start_idx = idx
        for _ in 1:spec.length
            value = evaluated[idx]
            push!(lower_hits, value < spec.lower)
            push!(upper_hits, value > spec.upper)
            push!(lipschitz_hits, false)
            evaluated[idx] = clamp(value, spec.lower, spec.upper)
            idx += 1
        end
        if spec.kind == :temporal
            for position in 2:spec.length
                current_idx = start_idx + position - 1
                previous_idx = current_idx - 1
                limited = clamp(
                    evaluated[current_idx],
                    evaluated[previous_idx] - MODULATION_LIPSCHITZ_DELTA,
                    evaluated[previous_idx] + MODULATION_LIPSCHITZ_DELTA,
                )
                if limited != evaluated[current_idx]
                    lipschitz_hits[current_idx] = true
                    evaluated[current_idx] = clamp(limited, spec.lower, spec.upper)
                end
            end
        end
    end
    return evaluated, Dict(
        "clipped" => any(lower_hits) || any(upper_hits) || any(lipschitz_hits),
        "lower_bound_hits" => lower_hits,
        "upper_bound_hits" => upper_hits,
        "lipschitz_hits" => lipschitz_hits,
    )
end

function cma_diagnostics(state::CMAState)
    values = eigen(Symmetric(state.covariance)).values
    values = max.(Float64.(values), 1e-12)
    return Dict(
        "eigenvalues" => values,
        "covariance_trace" => tr(state.covariance),
        "covariance_condition_number" => maximum(values) / minimum(values),
        "sigma_min" => minimum(state.sigma),
        "sigma_max" => maximum(state.sigma),
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
    y_w = (new_mean .- state.mean) ./ max.(state.sigma, 1e-12)
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
        y = (x .- state.mean) ./ max.(state.sigma, 1e-12)
        rank_mu_cov .+= w .* (y * y')
    end
    correction = (1.0 - h_sigma) * c_c * (2.0 - c_c)
    new_cov = (1.0 - c_1 - c_mu) .* state.covariance .+
              c_1 .* (p_c * p_c' + correction .* state.covariance) .+
              c_mu .* rank_mu_cov
    new_cov = 0.5 .* (new_cov + new_cov')
    # Coordinate-wise step-size adaptation. This deliberately replaces the
    # single global sigma so transferred posterior uncertainty is preserved
    # per parameter dimension.
    chi_1 = sqrt(2.0 / pi)
    new_sigma = state.sigma .* exp.((c_sigma / d_sigma) .* (abs.(p_sigma) ./ chi_1 .- 1.0))
    return CMAState(new_mean, clamp.(new_sigma, CMA_SIGMA_MIN, CMA_SIGMA_MAX), new_cov, p_c, p_sigma)
end

function safe_save_json(path::String, value; label::String=path)
    try
        save_json(path, value)
    catch err
        @error "Failed to save JSON artifact" label path err
        rethrow(err)
    end
end

function atomic_save_json(path::String, value; label::String=path)
    mkpath(dirname(path))
    tmp = tempname(dirname(path))
    try
        save_json(tmp, value)
        mv(tmp, path; force=true)
    catch err
        isfile(tmp) && rm(tmp)
        @error "Failed to atomically save JSON artifact" label path err
        rethrow(err)
    end
    return path
end

"""Persist terminal posterior rejection evidence and invalidate stale state.

The reusable-state path is replaced with an invalidation record using a
same-directory temporary file and rename, so a reader cannot observe a
partially written posterior state.  The invalidated record intentionally
omits state fields such as `mean` and `covariance`; `load_full_reusable_state`
therefore fails closed if a resumed stage encounters it.
"""
function persist_posterior_rejection!(
    stage_root::String,
    transition::AbstractDict;
    reusable_state_path::String=joinpath(stage_root, "posterior_reusable_state.json"),
    stage::String="",
    fit_months::Int=0,
)
    evidence = deepcopy(get(transition, "terminal_evidence", Dict{String,Any}()))
    evidence["status"] = "failed"
    evidence["policy_outcome"] = "reject"
    evidence["failure_class"] = get(
        evidence, "failure_class", "posterior_transition_policy_reject",
    )
    evidence["stage"] = stage
    evidence["fit_months"] = fit_months
    evidence["terminal"] = true
    rejection_path = joinpath(stage_root, "posterior_transition_rejected.json")
    safe_save_json(rejection_path, evidence; label="posterior_transition_rejected")

    invalidated = Dict{String,Any}(
        "status" => "invalidated",
        "policy_outcome" => "reject",
        "failure_class" => evidence["failure_class"],
        "stage" => stage,
        "fit_months" => fit_months,
        "terminal_evidence_path" => rejection_path,
        "invalidated_at" => string(Dates.now()),
    )
    if isfile(reusable_state_path)
        mkpath(dirname(reusable_state_path))
        temporary_path = tempname(dirname(reusable_state_path))
        try
            save_json(temporary_path, invalidated)
            mv(temporary_path, reusable_state_path; force=true)
        finally
            isfile(temporary_path) && rm(temporary_path; force=true)
        end
    end

    blocked = Dict{String,Any}(
        "status" => "blocked",
        "terminal" => true,
        "stage" => stage,
        "fit_months" => fit_months,
        "policy_outcome" => "reject",
        "failure_class" => evidence["failure_class"],
        "reason" => "posterior transfer rejected; stage progression stopped",
        "posterior_transition_rejected" => rejection_path,
        "invalidated_reusable_state" => reusable_state_path,
        "next_stage_created" => false,
    )
    safe_save_json(joinpath(stage_root, "stage_blocked.json"), blocked; label="stage_blocked")
    return blocked
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
    metrics::AbstractDict,
    metrics_path::String,
    parameter_names::Vector{String},
    transition_report::Union{Nothing,AbstractDict}=nothing,
)
    metric_values = haskey(metrics, "metrics") && metrics["metrics"] isa AbstractDict ?
        metrics["metrics"] :
        metrics
    append_jsonl(joinpath(iter_root, "posterior_training_data.jsonl"), Dict(
        "schema_version" => "experiment-v1",
        "experiment_type" => "cma_candidate",
        "stage" => stage.name,
        "iteration" => iteration,
        "candidate" => candidate_id,
        "parameter_names" => parameter_names,
        "x_raw" => raw_candidate,
        "x_evaluated" => evaluated_candidate,
        "z" => z,
        "clipping" => clip_info,
        "score" => score,
        "vector_log_likelihood" => get(metric_values, "vector_log_likelihood", -Float64(score)),
        "metrics_path" => metrics_path,
        "transition_delta_report" => transition_report,
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
    seed = seed_config === nothing && !isempty(cfg.runtime_seed) ?
        deepcopy(cfg.runtime_seed) : load_json(seed_path)
    seed_config === nothing && _normalize_seed_paths!(seed, dirname(seed_path))
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
function run_stage(
    rng::AbstractRNG,
    seed::Dict{String,Any},
    specs::Vector{ParamSpec},
    cfg::OptimizerConfig,
    stage::StageConfig,
    state::Union{Nothing,CMAState};
    use_slurm::Bool=false,
    resume_from::Int=0,
    previous_specs::Union{Nothing,Vector{ParamSpec}}=nothing,
    predecessor_archive=Any[],
)
    active_months = stage.fit_months
    days = active_months * cfg.monthly_days
    policy = determine_search_policy(cfg, stage)
    specs_stage = stage_specs(seed, specs, cfg, stage)
    dim = sum(spec.length for spec in specs_stage)
    stage_root = joinpath(cfg.output_dir, "real_sims", stage.name)
    resume_state = load_stage_state(stage_root)
    if resume_state !== nothing
        committed_check = validate_committed_artifacts(stage_root, "production-v1")
        committed_check["valid"] ||
            throw(ArgumentError("committed stage artifacts failed validation: " *
                                join(String.(committed_check["contradictions"]), "; ")))
    end
    if resume_state !== nothing && haskey(resume_state, "rng_state")
        # Restore before reusable-state construction or unlock logic can
        # consume a different stream position.
        copy!(rng, restore_rng(resume_state["rng_state"]))
    end
    received_prior = state !== nothing
    trusted_predecessor = nothing
    if !received_prior
        trusted_predecessor = load_immediate_predecessor_state(cfg, stage)
        if trusted_predecessor !== nothing
            # This branch is deliberately before initial-state construction:
            # a predecessor existing on disk makes the initial seed invalid as
            # an extension source in a fresh process.
            predecessor_archive = trusted_predecessor["archive"]
            trusted_seed = archive_entry_config(predecessor_archive)
            trusted_seed === nothing && throw(ArgumentError(
                "trusted predecessor archive has no reusable candidate configuration"))
            seed = trusted_seed
            reusable = trusted_predecessor["reusable_state"]
            state = build_state_from_reusable(
                seed,
                specs_stage,
                reusable;
                sigma_floor=max(0.08, 0.75 * stage.sigma),
                temporal_unlock_multiplier=policy.temporal_unlock_multiplier,
                covariance_inflation=cfg.posterior.transfer_covariance_inflation,
                sigma_multiplier=cfg.posterior.transfer_sigma_multiplier,
                new_dimension_variance=cfg.posterior.new_dimension_variance,
                rng=rng,
            )
            predecessor_state = trusted_predecessor["stage_state"]
            index = findfirst(s -> s.name == stage.name, cfg.stages)
            previous_specs === nothing && (previous_specs =
                stage_specs(seed, specs, cfg, cfg.stages[index - 1]))
            top_candidates = latest_iteration_top_candidates(trusted_predecessor["root"])
            state = temporal_unlock_from_bucket_errors!(state, specs_stage, top_candidates; rng=rng)
            received_prior = true
        end
    end
    if state === nothing
        x0 = initial_vector(seed, specs_stage)
        reusable_path = joinpath(cfg.output_dir, "full_reusable_state.json")
        reusable = load_full_reusable_state(reusable_path)
        if reusable !== nothing
            state = build_state_from_reusable(
                seed,
                specs_stage,
                reusable;
                sigma_floor=max(0.08, 0.75 * stage.sigma),
                temporal_unlock_multiplier=policy.temporal_unlock_multiplier,
                covariance_inflation=cfg.posterior.transfer_covariance_inflation,
                sigma_multiplier=cfg.posterior.transfer_sigma_multiplier,
                new_dimension_variance=cfg.posterior.new_dimension_variance,
                rng=rng,
            )
            top_candidates = latest_iteration_top_candidates(stage_root)
            state = temporal_unlock_from_bucket_errors!(state, specs_stage, top_candidates; rng=rng)
        else
            seeded = initial_state_from_config(cfg, dim, x0)
            state = seeded === nothing ? CMAState(copy(x0), stage.sigma, Matrix{Float64}(I, dim, dim)) : seeded
        end
    elseif trusted_predecessor === nothing
        state = stage_transition_state(
            state,
            stage,
            specs_stage;
            previous_specs=previous_specs,
            sigma_scale=1.25,
            sigma_floor=0.02,
        )
    end
    state = CMAState(
        copy(state.mean),
        clamp.(state.sigma * policy.sigma_multiplier, CMA_SIGMA_MIN, CMA_SIGMA_MAX),
        copy(state.covariance),
        copy(state.p_c),
        copy(state.p_sigma),
    )

    safe_save_json(joinpath(stage_root, "experiment_manifest.json"), Dict(
        "schema_version" => "experiment-v1",
        "experiment_type" => "cma_stage",
        "stage" => stage.name,
        "fit_months" => active_months,
        "monthly_days" => cfg.monthly_days,
        "simulation_days" => days,
        "parameter_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
        "population_size" => stage.population_size,
        "max_iterations" => stage.max_iterations,
        "early_stop" => Dict(
            "min_completion_fraction" => cfg.objective.min_completion_fraction,
            "finish_iter_delay" => cfg.objective.finish_iter_delay,
        ),
        "weekly_control_metrics" => WEEKLY_CONTROL_METRICS,
        "vector_likelihood" => cfg.posterior.likelihood,
        "survivor_archive" => Dict(
            "selection" => "adaptive_score_threshold_pareto_parameter_clusters",
            "max_size" => SURVIVOR_ARCHIVE_SIZE,
            "score_mad_multiplier" => SURVIVOR_SCORE_MAD_MULTIPLIER,
            "relative_score_floor" => SURVIVOR_RELATIVE_SCORE_FLOOR,
            "min_parameter_distance" => SURVIVOR_MIN_DISTANCE,
        ),
    ); label="experiment_manifest")
    history = Any[]
    iter_log = Any[]
    top_candidates = Dict{String,Dict{String,Any}}()
    best_score_raw = resume_state === nothing ? Inf : get(resume_state, "best_score", Inf)
    best_score = best_score_raw === nothing ? Inf : Float64(best_score_raw)
    best_vector = resume_state !== nothing && haskey(resume_state, "best_vector") ? Float64.(resume_state["best_vector"]) : copy(state.mean)
    best_candidate = deepcopy(seed)
    archive_path = joinpath(stage_root, "survivor_archive.json")
    survivor_archive = isfile(archive_path) ? load_json(archive_path) : Any[]
    survivor_archive isa AbstractVector || (survivor_archive = Any[])
    transfer_archive = received_prior ? deepcopy(predecessor_archive) : Any[]
    if resume_state !== nothing && haskey(resume_state, "best_vector")
        sigma_raw = get(resume_state, "sigma", state.sigma)
        sigma_resume = sigma_raw isa AbstractVector ?
            Float64.(sigma_raw) :
            fill(Float64(sigma_raw), dim)
        p_c = haskey(resume_state, "p_c") ? Float64.(resume_state["p_c"]) : zeros(dim)
        p_sigma = haskey(resume_state, "p_sigma") ? Float64.(resume_state["p_sigma"]) : zeros(dim)
        state = CMAState(copy(best_vector), sigma_resume, state.covariance, p_c, p_sigma)
    end
    start_iter = max(1, resume_from + 1)
    @info "Starting stage run" stage=stage.name fit_months=active_months start_iter=start_iter max_iterations=stage.max_iterations population_size=stage.population_size use_slurm=use_slurm search_policy=policy.name

    for iter in start_iter:stage.max_iterations
        @info "Starting iteration" stage=stage.name iteration=iter sigma=state.sigma best_score=best_score
        candidates, zs = cma_candidates(rng, state, stage.population_size)
        # Carry multiple plausible trajectories into the next stage instead
        # of transferring only the single best candidate.
        for (candidate_id, entry) in enumerate(transfer_archive[1:min(length(transfer_archive), length(candidates))])
            candidates[candidate_id] = archive_vector_for_stage(entry, specs_stage, candidates[candidate_id])
            zs[candidate_id] = zeros(length(state.mean))
        end
        immigrant_fraction = received_prior ?
            max(policy.random_candidate_fraction, cfg.posterior.immigrant_fraction) :
            policy.random_candidate_fraction
        if immigrant_fraction > 0.0
            # Transfer slots are protected: immigrants may only occupy the
            # disjoint suffix of the population.
            immigrant_start = min(length(transfer_archive), length(candidates)) + 1
            if immigrant_start <= length(candidates)
                for candidate_id in immigrant_start:length(candidates)
                    rand(rng) > immigrant_fraction && continue
                    idx = 1
                    for spec in specs_stage
                        for _ in 1:spec.length
                            candidates[candidate_id][idx] = spec.lower + rand(rng) * (spec.upper - spec.lower)
                            idx += 1
                        end
                    end
                end
            end
        end
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
                    inject_temporal_prefix_locks!(candidate_cfg, seed, previous_specs)
                    transfer_entry = ci <= length(transfer_archive) ? transfer_archive[ci] : nothing
                    candidate_class = transfer_entry === nothing ? "immigrant/escape" : "archive_transfer"
                    if transfer_entry !== nothing
                        x = archive_vector_for_stage(transfer_entry, specs_stage, x)
                        x, clip_info = clip_candidate(x, specs_stage)
                        cand_cfg = vector_to_config(seed, specs_stage, x, active_months)
                        inject_frozen!(cand_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                        inject_temporal_prefix_locks!(cand_cfg, seed, previous_specs)
                    end
                    transition = enforce_transition_policy(seed, candidate_cfg, specs_stage, transfer_entry;
                        candidate_class=candidate_class, policy="reject")
                    cand_dir = joinpath(iter_root, @sprintf("cand_%02d", ci))
                    mkpath(cand_dir)
                    if transition["status"] == "rejected"
                        safe_save_json(joinpath(cand_dir, "transition_rejected.json"),
                            transition["report"]; label="transition_rejected")
                        materialize_terminal_candidate!(cand_dir, "failed";
                            failure_class="transition_policy_reject",
                            details=Dict("stage" => stage.name, "iteration" => iter,
                                         "candidate" => ci,
                                         "transition_delta_report" => transition["report"]))
                    else
                        candidate_cfg = transition["config"]
                        save_json(joinpath(cand_dir, "config.json"), candidate_cfg)
                        println(io, cand_dir)
                    end
                end
            end
            jobid = submit_slurm_array(cfg, list_file)
            @info "Submitted Slurm array" stage=stage.name iteration=iter jobid=jobid
            wait_result = wait_for_iteration_outputs(
                list_file;
                poll=10.0,
                min_completion_fraction=cfg.objective.min_completion_fraction,
                finish_iter_delay=cfg.objective.finish_iter_delay,
                max_wait=Float64(get(cfg.validation, "iteration_timeout_seconds",
                                     get(cfg.validation, "adapter_timeout_seconds", 3600.0) + 300.0)),
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
                inject_temporal_prefix_locks!(cand_cfg, seed, previous_specs)
                transfer_entry = ci <= length(transfer_archive) ? transfer_archive[ci] : nothing
                candidate_class = transfer_entry === nothing ? "immigrant/escape" : "archive_transfer"
                if transfer_entry !== nothing
                    x = archive_vector_for_stage(transfer_entry, specs_stage, x)
                    x, clip_info = clip_candidate(x, specs_stage)
                    cand_cfg = vector_to_config(seed, specs_stage, x, active_months)
                    inject_frozen!(cand_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                    inject_temporal_prefix_locks!(cand_cfg, seed, previous_specs)
                end
                transition = enforce_transition_policy(seed, cand_cfg, specs_stage, transfer_entry;
                    candidate_class=candidate_class, policy="reject")
                transition_report = transition["report"]
                cand_cfg = transition["config"]
                transition_rejected = transition["status"] == "rejected"
                skipped = isfile(joinpath(cand_dir, "skipped.ok"))
                metrics = if transition_rejected
                    Dict("score" => Inf, "metrics" => Dict(), "simulated" => "transition_rejected",
                         "status" => "failed", "failure_class" => "transition_policy_reject")
                elseif skipped
                    metrics_payload = Dict("score" => Inf, "simulated" => "real_skipped", "status" => "skipped")
                    safe_save_json(joinpath(cand_dir, "metrics.json"), metrics_payload; label="candidate_metrics")
                    Dict("score" => Inf, "metrics" => Dict(), "simulated" => "real_skipped", "status" => "skipped")
                elseif isfile(daily_path)
                    combined, comp = score_from_daily(cfg, daily_path, days, cand_cfg)
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
                    vector_likelihood = vector_likelihood_payload(
                        daily_path, gt, days; family=cfg.posterior.likelihood,
                        metric_names=likelihood_metric_names(cfg),
                        dispersions=likelihood_dispersions(cfg)
                    )
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
                        "schema_version" => "experiment-v1",
                        "experiment_type" => "cma_candidate",
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
                        "vector_likelihood" => vector_likelihood,
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
                        state, score, metrics, joinpath(cand_dir, "metrics.json"),
                        ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
                        transition_report
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
                    "evaluated_vector" => copy(x),
                    "parameter_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
                    "config" => deepcopy(cand_cfg),
                    "metrics" => metrics,
                    "transition_delta_report" => transition_report,
                    "provenance" => Dict{String,Any}(
                        "source" => transfer_entry === nothing ? "cma_population" : "predecessor_archive",
                        "predecessor_archive_entry" => transfer_entry === nothing ? nothing : get(transfer_entry, "candidate", nothing),
                        "candidate_class" => candidate_class,
                    ),
                )
                top_candidates[key] = candidate_entry
                push!(iteration_top_candidates, candidate_entry)
                if get(metrics, "status", "failed") == "completed"
                    push!(ranked, (score, x, zs[ci]))
                end
                if get(metrics, "status", "failed") == "completed" && score < best_score
                    best_score = score
                    best_vector = copy(x)
                    best_candidate = deepcopy(cand_cfg)
                end
            end
        else
            local_candidate_dirs = [
                joinpath(iter_root, @sprintf("cand_%02d", candidate_id))
                for candidate_id in 1:length(candidates)
            ]
            local_iter_records = Any[]
            mkpath.(local_candidate_dirs)
            open(joinpath(iter_root, "candidate_list.txt"), "w") do io
                foreach(d -> println(io, d), local_candidate_dirs)
            end
            for (ci, cand) in enumerate(candidates)
                iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
                mkpath(iter_root)
                cand_dir = local_candidate_dirs[ci]
                x, clip_info = clip_candidate(cand, specs_stage)
                candidate_cfg = vector_to_config(seed, specs_stage, x, active_months)
                inject_frozen!(candidate_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                inject_temporal_prefix_locks!(candidate_cfg, seed, previous_specs)
                transfer_entry = ci <= length(transfer_archive) ? transfer_archive[ci] : nothing
                candidate_class = transfer_entry === nothing ? "immigrant/escape" : "archive_transfer"
                if transfer_entry !== nothing
                    x = archive_vector_for_stage(transfer_entry, specs_stage, x)
                    x, clip_info = clip_candidate(x, specs_stage)
                    cand_cfg = vector_to_config(seed, specs_stage, x, active_months)
                    inject_frozen!(cand_cfg, seed, specs, get(cfg.stage_freeze, stage.name, String[]))
                    inject_temporal_prefix_locks!(cand_cfg, seed, previous_specs)
                end
                transition = enforce_transition_policy(seed, candidate_cfg, specs_stage, transfer_entry;
                    candidate_class=candidate_class, policy="reject")
                transition_report = transition["report"]
                candidate_cfg = transition["config"]
                existing_terminal = candidate_terminal_status(cand_dir)
                # A caller-created skipped marker is authoritative.  In
                # particular, local scoring must not turn an explicitly
                # skipped candidate back into a completed one.
                metrics = existing_terminal["status"] == "skipped" ?
                    Dict{String,Any}("score" => Inf, "status" => "skipped",
                                     "simulated" => "real_skipped",
                                     "failure_class" => "iteration_truncated") :
                    existing_terminal["status"] == "failed" ?
                    Dict{String,Any}("score" => Inf, "status" => "failed",
                                     "simulated" => "real_failed",
                                     "failure_class" => get(existing_terminal, "failure_class", "adapter_failure")) :
                    transition["status"] == "rejected" ?
                    Dict{String,Any}("score" => Inf, "status" => "failed",
                                     "simulated" => "transition_rejected",
                                     "failure_class" => "transition_policy_reject") :
                    score_candidate(candidate_cfg, cfg, days; workdir=cand_dir)
                if existing_terminal["status"] == "skipped"
                    _materialize_terminal_artifact!(cand_dir, existing_terminal)
                elseif existing_terminal["status"] == "failed"
                    _materialize_terminal_artifact!(cand_dir, existing_terminal)
                elseif transition["status"] == "rejected"
                    materialize_terminal_candidate!(cand_dir, "failed";
                        failure_class="transition_policy_reject",
                        details=Dict("stage" => stage.name, "iteration" => iter,
                                     "candidate" => ci,
                                     "transition_delta_report" => transition_report))
                else
                    safe_save_json(joinpath(cand_dir, "metrics.json"), metrics; label="candidate_metrics")
                    materialize_terminal_candidate!(cand_dir,
                        get(metrics, "status", "failed") == "completed" ? "completed" : "failed";
                        failure_class=get(metrics, "status", "completed") == "completed" ?
                            nothing : get(metrics, "failure_class", "adapter_failure"),
                        details=Dict("stage" => stage.name, "iteration" => iter, "candidate" => ci))
                end
                score = metrics["score"]
                if get(metrics, "status", "completed") == "completed" && isfinite(Float64(score))
                    append_cma_candidate_record(
                        iter_root, stage, iter, ci, cand, x, zs[ci], clip_info,
                        state, score, metrics, joinpath(cand_dir, "metrics.json"),
                        ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
                        transition_report
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
                    "status" => get(metrics, "status", "unknown"),
                    "config" => deepcopy(candidate_cfg),
                    "evaluated_vector" => copy(x),
                    "parameter_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
                    "metrics" => metrics,
                    "transition_delta_report" => transition_report,
                    "provenance" => Dict{String,Any}(
                        "source" => transfer_entry === nothing ? "cma_population" : "predecessor_archive",
                        "predecessor_archive_entry" => transfer_entry === nothing ? nothing : get(transfer_entry, "candidate", nothing),
                        "candidate_class" => candidate_class,
                    ),
                )
                top_candidates[key] = candidate_entry
                push!(iteration_top_candidates, candidate_entry)
                if get(metrics, "status", "failed") == "completed" && isfinite(Float64(score))
                    push!(ranked, (score, x, zs[ci]))
                end
                if get(metrics, "status", "failed") == "completed" && score < best_score
                    best_score = score
                    best_vector = copy(x)
                    best_candidate = deepcopy(candidate_cfg)
                end
                push!(local_iter_records, Dict(
                    "stage" => stage.name, "iteration" => iter, "candidate" => ci,
                    "status" => get(metrics, "status", "unknown"),
                    "score" => score,
                    "threshold_reached" => false,
                    "iteration_truncated" => false,
                    "completed_count" => 0,
                    "failed_count" => 0,
                    "pending_count" => 0,
                ))
            end
            # Account only after every local candidate has a terminal marker.
            # This makes local threshold/truncation fields match the
            # collected/Slurm normalized result rather than an early snapshot.
            local_wait_result = normalized_iteration_result(local_candidate_dirs;
                min_completion_fraction=cfg.objective.min_completion_fraction)
            for record in local_iter_records
                record["threshold_reached"] = local_wait_result["threshold_reached"]
                record["iteration_truncated"] = local_wait_result["iteration_truncated"]
                record["completed_count"] = local_wait_result["done"]
                record["failed_count"] = local_wait_result["failed"]
                record["skipped_count"] = local_wait_result["skipped"]
                record["pending_count"] = local_wait_result["pending_count"]
                append_jsonl(joinpath(stage_root, "iter_metrics.jsonl"), record)
            end
        end
        isempty(ranked) && error("No completed candidates available for stage $(stage.name) iteration $(iter). Increase max wait or completion fraction.")
        sort!(ranked, by=first)
        @info "Updating CMA state" stage=stage.name iteration=iter completed_candidates=length(ranked) best_iteration_score=ranked[1][1]
        state = update_state(state, ranked)
        push!(iter_log, Dict(
            "stage" => stage.name,
            "iteration" => iter,
            "param_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
            "search_policy" => policy.name,
            "best_score" => best_score,
            "sigma" => state.sigma,
            "covariance_trace" => tr(state.covariance),
        ))
        stage_transition_report = effective_transition_report(seed, Dict(
            "values" => state.mean,
            "param_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
        ), specs_stage, isempty(transfer_archive) ? nothing : transfer_archive[1];
            candidate_class=isempty(transfer_archive) ? "new_dimension" : "archive_transfer")
        trusted_values = Float64.(state.mean)
        trusted_names = ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length]
        trusted_trajectory = Dict{String,Any}(
            "identity" => bytes2hex(SHA.sha256("production-trusted-trajectory-v1")),
            "values" => trusted_values,
            "prefix_values" => trusted_values,
        )
        trusted_prefix_hash = bytes2hex(SHA.sha256(JSON.json(trusted_trajectory["prefix_values"])))
        trusted_locks = [Dict{String,Any}(
            "name" => trusted_names[i], "start_day" => i, "end_day" => i,
            "value" => trusted_values[i], "class" => "locked")
            for i in eachindex(trusted_values)]
        nested_cma = Dict{String,Any}(
            "parameter_names" => trusted_names, "mean" => state.mean,
            "sigma" => state.sigma, "covariance" => state.covariance,
            "p_c" => state.p_c, "p_sigma" => state.p_sigma)
        safe_save_json(joinpath(stage_root, "stage_state.json"), Dict(
            "stage" => stage.name,
            "iteration" => iter,
            "param_names" => ["$(spec.name)[$i]" for spec in specs_stage for i in 1:spec.length],
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
            "rng_state" => rng_snapshot(rng),
            "transition_delta_report" => stage_transition_report,
            "historical_trajectory" => trusted_trajectory,
            "trajectory_identity" => trusted_trajectory["identity"],
            "prefix_hash" => trusted_prefix_hash,
            "locked_intervals" => trusted_locks,
            "cma_state" => nested_cma,
        ); label="stage_state")
        iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_$(iter)")
        mkpath(iter_root)
        iteration_top_payload = safe_iteration_top_k(iteration_top_candidates, cfg.objective.top_k)
        safe_save_json(joinpath(iter_root, "top_candidates.json"), iteration_top_payload; label="iteration_top_candidates")
        # Keep a canonical stage-level copy for validators and fresh-process
        # resume.  It is replaced only after the iteration has been fully
        # evaluated and is covered by the commit hash manifest.
        safe_save_json(joinpath(stage_root, "top_candidates.json"), iteration_top_payload; label="stage_top_candidates")
        archive_report = survivor_archive_update(
            survivor_archive, iteration_top_candidates;
            current_stage=stage.name, current_fit_months=active_months,
            target_size=40, max_size=SURVIVOR_ARCHIVE_SIZE,
            return_report=true,
        )
        survivor_archive = archive_report["archive"]
        safe_save_json(archive_path, survivor_archive; label="survivor_archive")
        persist_archive_transfer_manifest(stage_root, survivor_archive;
            archive_path=archive_path, stage=stage.name, fit_months=active_months)
        safe_save_json(joinpath(stage_root, "survivor_archive_summary.json"),
            archive_report; label="survivor_archive_summary")
        reusable_payload = full_reusable_state_from_cma(stage, specs_stage, state;
            transition_report=stage_transition_report)
        archive_ids = [get(entry, "candidate", nothing) for entry in survivor_archive]
        reusable_payload["historical_trajectory"] = trusted_trajectory
        reusable_payload["trajectory_identity"] = trusted_trajectory["identity"]
        reusable_payload["prefix_hash"] = trusted_prefix_hash
        reusable_payload["locked_intervals"] = trusted_locks
        reusable_payload["cma_state"] = nested_cma
        reusable_payload["archive_ids"] = archive_ids
        reusable_payload["selected_archive_ids"] = archive_ids
        reusable_payload["archive_lineage"] = Dict(
            "canonical_archive_path" => abspath(archive_path),
            "archive_ids" => archive_ids,
            "source_stage" => stage.name,
            "fit_months" => active_months)
        safe_save_json(joinpath(stage_root, "full_reusable_state.json"),
            reusable_payload; label="full_reusable_state")
        # The archive IDs are only known after selection.  Update the stage
        # state before hashing the commit so both trusted state files carry
        # the same admitted set and lineage.
        committed_stage_state = load_json(joinpath(stage_root, "stage_state.json"))
        committed_stage_state["archive_ids"] = archive_ids
        committed_stage_state["archive_path"] = abspath(archive_path)
        committed_stage_state["archive_lineage"] = reusable_payload["archive_lineage"]
        safe_save_json(joinpath(stage_root, "stage_state.json"),
            committed_stage_state; label="stage_state_lineage")
        # This is the sole resume authority for an iteration.  It is written
        # last, after every cross-file artifact, and contains identity fields
        # so a stale manifest can never make another iteration look complete.
        committed_artifacts = [
            "stage_state.json", "iter_metrics.jsonl", "top_candidates.json",
            "survivor_archive.json", "survivor_archive_summary.json",
            "full_reusable_state.json", "archive_transfer_manifest.json",
            joinpath("iter_$(iter)", "candidate_list.txt"),
            joinpath("iter_$(iter)", "cma_sampling_state.json"),
            joinpath("iter_$(iter)", "top_candidates.json"),
        ]
        artifact_hashes = Dict{String,Any}(
            relative => bytes2hex(SHA.sha256(read(joinpath(stage_root, relative))))
            for relative in committed_artifacts
        )
        artifact_digest = bytes2hex(SHA.sha256(JSON.json(artifact_hashes)))
        atomic_save_json(joinpath(iter_root, "iteration_commit.json"), Dict(
            "status" => "committed",
            "schema_version" => "production-v1",
            "stage" => stage.name,
            "iteration" => iter,
            "artifact_key_set" => committed_artifacts,
            "artifact_hashes" => artifact_hashes,
            "artifact_hash_manifest" => artifact_digest,
            "artifact_integrity_digest" => artifact_digest,
        ); label="iteration_commit")
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
        "survivor_archive" => survivor_archive,
        "archive_summary" => isfile(joinpath(stage_root, "survivor_archive_summary.json")) ?
            load_json(joinpath(stage_root, "survivor_archive_summary.json")) : nothing,
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
    seed = isempty(cfg.runtime_seed) ? load_json(cfg.seed_config) : deepcopy(cfg.runtime_seed)
    specs = build_specs(seed, cfg)
    rng = MersenneTwister(42)

    stage_outputs = Any[]
    all_history = Any[]
    state = nothing
    previous_specs = nothing
    current_seed = deepcopy(seed)
    predecessor_archive = Any[]

    for (stage_index, stage) in enumerate(cfg.stages)
        stage_root = joinpath(cfg.output_dir, "real_sims", stage.name)
        if stage_index > 1
            predecessor_root = joinpath(cfg.output_dir, "real_sims", cfg.stages[stage_index - 1].name)
            predecessor_check = validate_committed_artifacts(predecessor_root, "production-v1")
            predecessor_check["valid"] ||
                throw(ArgumentError("predecessor stage artifacts failed validation: " *
                                    join(String.(predecessor_check["contradictions"]), "; ")))
        end
        resume_info = stage_resume_info(stage_root)
        resume_from = if resume_info === nothing
            0
        elseif Bool(get(resume_info, "iteration_completed", false))
            Int(resume_info["last_iter"])
        else
            # An existing candidate directory is recoverable work, not a
            # committed iteration. Re-enter that iteration rather than
            # silently advancing past nonterminal candidates.
            max(Int(get(resume_info, "resume_iteration", 1)) - 1, 0)
        end
        if resume_info !== nothing
            @info "Resuming stage from artifacts" stage=stage.name resume_from=resume_from
        end
        result, state = run_stage(
            rng,
            current_seed,
            specs,
            cfg,
            stage,
            state;
            use_slurm=use_slurm,
            resume_from=resume_from,
            previous_specs=previous_specs,
            predecessor_archive=predecessor_archive,
        )
        if stage_index < length(cfg.stages)
            archive_summary = get(result, "archive_summary", nothing)
            archive_values = get(result, "survivor_archive", Any[])
            archive_summary isa AbstractDict || (archive_summary = Dict{String,Any}())
            gate = archive_quality_gate(archive_values;
                current_stage=stage.name, current_fit_months=stage.fit_months,
                current_best_score=result["best_score"], target_size=40,
                # These are deliberately external inputs.  The archive's
                # self-derived band/diversity report is descriptive only and
                # cannot make a poor current-stage objective pass.
                current_quality_band=get(cfg.validation, "current_quality_band", nothing),
                current_minimum_size=get(cfg.validation, "current_minimum_archive_size", nothing),
                current_diversity_passed=get(cfg.validation, "current_diversity_passed", nothing),
                quality_band_constrained=Bool(get(cfg.validation, "allow_quality_band_reduction", false)))
            safe_save_json(joinpath(stage_root, "stage_extension_gate.json"), gate;
                label="stage_extension_gate")
            if gate["status"] != "passed"
                blocked = merge(gate, Dict{String,Any}(
                    "status" => "blocked",
                    "next_stage_created" => false,
                    "reason" => gate["refusal_reason"],
                ))
                safe_save_json(joinpath(stage_root, "stage_blocked.json"), blocked;
                    label="stage_blocked")
                push!(stage_outputs, Dict(
                    "stage" => result["stage"], "best_score" => result["best_score"],
                    "archive_count" => length(archive_values), "extension_gate" => gate,
                ))
                append!(all_history, result["history"])
                break
            end
        end
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
                posterior_archive_path = joinpath(stage_root, "survivor_archive.json")
                posterior_archive = isfile(posterior_archive_path) ? load_json(posterior_archive_path) : Any[]
                posterior_archive isa AbstractVector || (posterior_archive = Any[])
                posterior_source = isempty(posterior_archive) ? nothing : posterior_archive[1]
                posterior_specs = stage_specs(current_seed, specs, cfg, stage)
                posterior_transition = enforce_posterior_reusable_state(
                    current_seed, posterior_specs, posterior_state, posterior_source;
                    active_months=stage.fit_months, policy="reject",
                )
                if posterior_transition["status"] == "rejected"
                    # Rejection is terminal for this stage.  Persist evidence,
                    # invalidate any stale reusable posterior atomically, and
                    # stop before archive seed derivation or next-stage setup.
                    persist_posterior_rejection!(
                        stage_root,
                        posterior_transition;
                        reusable_state_path=joinpath(stage_root, "posterior_reusable_state.json"),
                        stage=stage.name,
                        fit_months=stage.fit_months,
                    )
                    break
                else
                    posterior_state = posterior_transition["state"]
                    posterior_state["archive_provenance"] = posterior_source === nothing ? nothing :
                        Dict{String,Any}("source_archive_path" => posterior_archive_path,
                                         "source_archive_entry" => get(posterior_source, "candidate", nothing),
                                         "source_stage" => get(posterior_source, "stage", nothing),
                                         "predecessor" => "immediate_admitted_archive")
                    posterior_state["predecessor_provenance"] = posterior_source === nothing ? nothing :
                        Dict{String,Any}("archive_path" => posterior_archive_path,
                                         "archive_entry_id" => get(posterior_source, "candidate", nothing),
                                         "stage" => get(posterior_source, "stage", nothing),
                                         "horizon_months" => get(posterior_source, "fit_months", stage.fit_months))
                    state = build_state_from_reusable(
                        current_seed,
                        posterior_specs,
                        posterior_state;
                        sigma_floor=max(0.08, 0.75 * stage.sigma),
                        temporal_unlock_multiplier=1.0,
                        covariance_inflation=cfg.posterior.transfer_covariance_inflation,
                        sigma_multiplier=cfg.posterior.transfer_sigma_multiplier,
                        new_dimension_variance=cfg.posterior.new_dimension_variance,
                        rng=rng,
                    )
                    safe_save_json(
                        joinpath(stage_root, "posterior_reusable_state.json"),
                        posterior_state;
                        label="posterior_reusable_state",
                    )
                end
            end
        end
        if stage_index < length(cfg.stages)
            next_stage = cfg.stages[stage_index + 1]
            predecessor_manifest_path = joinpath(stage_root, "archive_transfer_manifest.json")
            predecessor_archive = load_transfer_survivor_archive(
                joinpath(cfg.output_dir, "real_sims"), next_stage.name;
                predecessor_stage=stage.name,
                expected_fit_months=stage.fit_months,
                expected_manifest_path=predecessor_manifest_path,
                stage_order=[s.name for s in cfg.stages],
            )
            archive_seed = archive_entry_config(predecessor_archive)
            # The admitted predecessor archive owns the trusted prefix.  The
            # scalar best remains reporting-only and cannot become the next seed.
            archive_seed !== nothing && (current_seed = archive_seed)
        end
        previous_specs = stage_specs(current_seed, specs, cfg, stage)
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
    validation_replicates = Dict{String,Any}("enabled" => false)
    if cfg.external_sim !== nothing && !isempty(cfg.stages)
        validation_days = cfg.stages[end].fit_months * cfg.monthly_days
        validation_replicates = run_validation_replicates(
            cfg,
            current_seed,
            validation_days;
            workdir=joinpath(cfg.output_dir, "validation_replicates"),
        )
        safe_save_json(
            joinpath(cfg.output_dir, "validation_replicates.json"),
            validation_replicates;
            label="validation_replicates",
        )
    end
    if cfg.external_sim !== nothing && !isempty(cfg.stages)
        plot_script = joinpath(MANAGER_ROOT, "scripts", "plot_best_modulation_detections.py")
        if isfile(plot_script)
            try
                python_bin = get(ENV, "PYTHON_BIN", "python3")
                stage_dir = joinpath(cfg.output_dir, "real_sims", cfg.stages[end].name)
                plot_output = joinpath(cfg.output_dir, "infection_modulation_best.png")
                run(`$python_bin $plot_script --stage-dir $stage_dir --gt-dir $(cfg.external_sim.gt_dir) --out $plot_output`)
            catch err
                @warn "Failed to generate final modulation plot" err
            end
        end
    end
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
    return Dict(
        "stage_summary" => stage_outputs,
        "output_dir" => cfg.output_dir,
        "top_k" => cfg.objective.top_k,
        "search_policy" => cfg.objective.search_policy,
        "validation_replicates" => validation_replicates,
    )
end

function main()
    use_slurm = "--slurm" in ARGS
    preflight_only = "--preflight" in ARGS || "--no-launch" in ARGS
    config_args = filter(arg -> !(arg in ("--slurm", "--preflight", "--no-launch")), ARGS)
    config_path = length(config_args) >= 1 ? config_args[1] : joinpath(dirname(@__DIR__), "optimizer_config.json")
    result = preflight_only ? preflight_config(config_path; readiness=true) :
             run_optimizer(config_path; use_slurm=use_slurm)
    println(JSON.json(result))
end

end
