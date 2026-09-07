function _git_commit(path::String)
    try
        return strip(read(`git -C $path rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

function _file_identity(path::String)
    return Dict{String,Any}(
        "path" => abspath(path), "bytes" => filesize(path),
        "sha256" => bytes2hex(open(sha256, path)))
end

"""Validate the minimum output contract of a real simulator daily JLD2 file."""
function validate_simulation_jld2(path::String;
                                  required_metrics::Vector{String}=[
                                      "daily_detections", "daily_deaths",
                                      "daily_hospitalizations"],
                                  minimum_days::Int=1)
    minimum_days > 0 || throw(ArgumentError("minimum_days must be positive"))
    isfile(path) || return Dict{String,Any}(
        "valid" => false, "path" => path, "errors" => ["missing output"])
    errors = String[]
    trajectories = Any[]
    metrics_seen = Set{String}()
    try
        HDF5.h5open(path, "r") do file
            isempty(keys(file)) && push!(errors, "output has no trajectory groups")
            for trajectory_name in sort!(String.(collect(keys(file))))
                group = file[trajectory_name]
                available = Set(String.(collect(keys(group))))
                union!(metrics_seen, available)
                dimensions = Dict{String,Any}()
                for metric in required_metrics
                    if !(metric in available)
                        push!(errors, "$trajectory_name is missing $metric")
                        continue
                    end
                    values = Float64.(vec(read(group[metric])))
                    isempty(values) && push!(errors, "$trajectory_name.$metric is empty")
                    length(values) >= minimum_days || push!(errors,
                        "$trajectory_name.$metric has fewer than $minimum_days days")
                    all(isfinite, values) ||
                        push!(errors, "$trajectory_name.$metric contains non-finite values")
                    dimensions[metric] = length(values)
                end
                push!(trajectories, Dict("name" => trajectory_name,
                                         "dimensions" => dimensions))
            end
        end
    catch err
        push!(errors, "unreadable JLD2/HDF5: $(sprint(showerror, err))")
    end
    return Dict{String,Any}(
        "valid" => isempty(errors), "path" => abspath(path), "errors" => errors,
        "required_metrics" => required_metrics,
        "minimum_days" => minimum_days,
        "metrics_seen" => sort!(collect(metrics_seen)),
        "trajectory_count" => length(trajectories), "trajectories" => trajectories,
        "identity" => isfile(path) ? _file_identity(path) : nothing)
end

function _smoke_input_identities(preflight::AbstractDict)
    identities = Dict{String,Any}()
    for (name, details) in preflight["path_identities"]
        get(details, "type", "") == "file" || continue
        identities[String(name)] = Dict(
            "path" => preflight["paths"][name], "sha256" => details["sha256"])
    end
    ground_truth = get(preflight, "ground_truth", Dict{String,Any}())
    protocol = get(ground_truth, "data_protocol", Dict{String,Any}())
    for (metric, details) in get(protocol, "metrics", Dict{String,Any}())
        get(details, "status", "") == "valid" || continue
        identities["ground_truth.$metric"] = Dict(
            "source" => details["source"], "sha256" => details["sha256"])
    end
    return identities
end

"""Run the two-candidate real-adapter production smoke contract."""
function run_production_smoke(config_path::String; use_slurm::Bool=false)
    preflight = preflight_config(config_path; readiness=true)
    cfg = load_config(config_path)
    length(cfg.stages) == 1 || throw(ArgumentError("smoke config must contain exactly one stage"))
    stage = only(cfg.stages)
    stage.max_iterations == 1 || throw(ArgumentError("smoke stage must run exactly one iteration"))
    stage.population_size == 2 || throw(ArgumentError("smoke stage must contain exactly two candidates"))
    started = now(UTC)
    result = nothing
    run_error = nothing
    try
        result = run_optimizer(config_path; use_slurm=use_slurm)
    catch err
        run_error = sprint(showerror, err)
    end
    iter_root = joinpath(cfg.output_dir, "real_sims", stage.name, "iter_1")
    candidates = Any[]
    for candidate_id in 1:2
        root = joinpath(iter_root, @sprintf("cand_%02d", candidate_id))
        output = joinpath(root, "output_daily.jld2")
        validation = validate_simulation_jld2(output;
            minimum_days=stage.fit_months * cfg.monthly_days)
        invocation_path = joinpath(root, "adapter_invocation.json")
        invocation = isfile(invocation_path) ? load_json(invocation_path) : nothing
        status = candidate_terminal_status(root)
        push!(candidates, Dict{String,Any}(
            "candidate" => candidate_id, "status" => status,
            "adapter_invocation" => invocation, "output_validation" => validation))
    end
    all_valid = run_error === nothing &&
                all(get(c["status"], "status", "") == "completed" &&
                    Bool(c["output_validation"]["valid"]) for c in candidates)
    manifest = Dict{String,Any}(
        "schema_version" => "production-smoke-v1",
        "status" => all_valid ? "passed" : "failed",
        "execution_mode" => use_slurm ? "slurm" : "local",
        "started_at" => string(started), "finished_at" => string(now(UTC)),
        "optimizer_commit" => _git_commit(MANAGER_ROOT),
        "launcher_commit" => _git_commit(cfg.external_sim.project_dir),
        "environment" => Dict(
            "julia_version" => string(VERSION), "kernel" => string(Sys.KERNEL),
            "architecture" => string(Sys.ARCH), "cpu_threads" => Sys.CPU_THREADS,
            "active_project" => Base.active_project(), "working_directory" => pwd()),
        "input_identities" => _smoke_input_identities(preflight),
        "rng" => Dict("optimizer_seed" => 42,
                      "validation_seeds" => get(cfg.validation, "seeds", Int[])),
        "stage" => Dict("name" => stage.name, "fit_months" => stage.fit_months,
                        "iterations" => 1, "population_size" => 2),
        "commands" => [c["adapter_invocation"] === nothing ? nothing :
                       c["adapter_invocation"]["command"] for c in candidates],
        "candidates" => candidates, "optimizer_result" => result,
        "run_error" => run_error)
    manifest_path = joinpath(cfg.output_dir, "production_smoke_manifest.json")
    safe_save_json(manifest_path, manifest; label="production_smoke_manifest")
    all_valid || error("production smoke failed; inspect $manifest_path")
    return manifest
end

"""Compare local and Slurm smoke contracts without requiring stochastic equality."""
function compare_smoke_manifests(local_path::String, slurm_path::String)
    local_manifest, slurm_manifest = load_json(local_path), load_json(slurm_path)
    contradictions = String[]
    get(local_manifest, "execution_mode", "") == "local" ||
        push!(contradictions, "first manifest is not local")
    get(slurm_manifest, "execution_mode", "") == "slurm" ||
        push!(contradictions, "second manifest is not Slurm")
    local_manifest["input_identities"] == slurm_manifest["input_identities"] ||
        push!(contradictions, "input identities differ")
    local_manifest["optimizer_commit"] == slurm_manifest["optimizer_commit"] ||
        push!(contradictions, "optimizer commits differ")
    local_manifest["launcher_commit"] == slurm_manifest["launcher_commit"] ||
        push!(contradictions, "launcher commits differ")
    local_manifest["stage"] == slurm_manifest["stage"] ||
        push!(contradictions, "stage contracts differ")
    for candidate_id in 1:2
        left = local_manifest["candidates"][candidate_id]["output_validation"]
        right = slurm_manifest["candidates"][candidate_id]["output_validation"]
        left["required_metrics"] == right["required_metrics"] ||
            push!(contradictions, "candidate $candidate_id required metrics differ")
        left["trajectory_count"] == right["trajectory_count"] ||
            push!(contradictions, "candidate $candidate_id trajectory counts differ")
        left["trajectories"] == right["trajectories"] ||
            push!(contradictions, "candidate $candidate_id output dimensions differ")
    end
    return Dict{String,Any}(
        "schema_version" => "production-smoke-parity-v1",
        "status" => isempty(contradictions) ? "passed" : "failed",
        "contradictions" => contradictions,
        "local_manifest" => abspath(local_path), "slurm_manifest" => abspath(slurm_path))
end
