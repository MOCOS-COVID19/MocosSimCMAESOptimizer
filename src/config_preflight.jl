const ADAPTER_FAILURE_CLASSES = (
    "process_failure", "timeout", "missing_output", "malformed_output",
    "empty_data", "nonfinite_data", "score_exception", "missing_summary",
)

function _preflight_path(base::String, value, field::String)
    value isa AbstractString || throw(ArgumentError("$field must be a path string"))
    p = isabspath(String(value)) ? String(value) : normpath(joinpath(base, String(value)))
    ispath(p) || throw(ArgumentError("$field does not exist: $p"))
    return p
end

function _walk_seed_paths!(out::Dict{String,String}, node, prefix::String, base::String)
    node isa AbstractDict || return
    for (key, value) in node
        name = String(key)
        path = isempty(prefix) ? name : "$prefix.$name"
        if value isa AbstractString &&
           (endswith(lowercase(name), "_path") || occursin("population", lowercase(name)) ||
            occursin("covimod", lowercase(name)) || occursin("immunity", lowercase(name)))
            p = isabspath(String(value)) ? String(value) : normpath(joinpath(base, String(value)))
            ispath(p) || throw(ArgumentError("seed.$path does not exist: $p"))
            out[path] = p
        elseif value isa AbstractDict
            _walk_seed_paths!(out, value, path, base)
        end
    end
end

function _validate_gt_dir(gt_dir::String)
    csvs = Dict{String,Any}()
    for file in readdir(gt_dir)
        endswith(lowercase(file), ".csv") || continue
        path = joinpath(gt_dir, file)
        rows = Tuple{Int,Float64}[]
        open(path) do io
            first = true
            for line in eachline(io)
                first && (first = false; continue)
                parts = split(line, ',')
                length(parts) >= 2 || continue
                day = try parse(Int, strip(parts[1])) catch; continue end
                value = try parse(Float64, strip(parts[2])) catch; continue end
                isfinite(value) && push!(rows, (day, value))
            end
        end
        days = first.(rows)
        length(unique(days)) == length(days) || throw(ArgumentError("ground_truth.$file has duplicate day labels"))
        all(>(0), days) || throw(ArgumentError("ground_truth.$file has nonpositive day labels"))
        csvs[file] = Dict("path"=>path, "observations"=>length(rows),
                          "days"=>days, "valid"=>!isempty(rows))
    end
    isempty(csvs) && throw(ArgumentError("ground_truth has no CSV files"))
    any(v["valid"] for v in values(csvs)) ||
        throw(ArgumentError("ground_truth has no parseable observations"))
    return csvs
end

function _path_hash(path::String)
    bytes2hex(open(sha256, path))
end

"""
    preflight_config(path; readiness=false)

Resolve and validate a configuration without creating its output root or
starting an adapter. The returned manifest is the provenance contract used by
readiness and pipeline callers.
"""
function preflight_config(path::String; readiness::Bool=false)
    config_path = abspath(path)
    isfile(config_path) || throw(ArgumentError("config does not exist: $config_path"))
    raw = load_json(config_path)
    raw isa AbstractDict || throw(ArgumentError("config must be an object"))
    base = dirname(config_path)
    for field in ("stages", "scalar_bounds", "temporal_bounds", "objective", "seed_config")
        haskey(raw, field) || throw(ArgumentError("missing required section: $field"))
    end
    stages = raw["stages"]
    stages isa AbstractVector || throw(ArgumentError("stages must be an array"))
    names = String[]
    dimensions = Dict{String,Any}()
    for (i, stage) in enumerate(stages)
        stage isa AbstractDict || throw(ArgumentError("stages[$i] must be an object"))
        for field in ("name", "fit_months", "max_iterations", "population_size", "sigma")
            haskey(stage, field) || throw(ArgumentError("stages[$i].$field is required"))
        end
        name = String(stage["name"])
        !isempty(name) && !(name in names) || throw(ArgumentError("stages[$i].name must be unique and nonempty"))
        push!(names, name)
        Int(stage["fit_months"]) > 0 || throw(ArgumentError("stages[$i].fit_months must be positive"))
        Int(stage["max_iterations"]) > 0 || throw(ArgumentError("stages[$i].max_iterations must be positive"))
        Int(stage["population_size"]) > 0 || throw(ArgumentError("stages[$i].population_size must be positive"))
        sigma = Float64(stage["sigma"])
        isfinite(sigma) && sigma > 0 || throw(ArgumentError("stages[$i].sigma must be finite and positive"))
        dimensions[name] = Dict("fit_months"=>Int(stage["fit_months"]),
                                "population_size"=>Int(stage["population_size"]))
    end
    function bounds(section, label)
        section isa AbstractDict || throw(ArgumentError("$label must be an object"))
        result = Dict{String,Any}()
        for (name, pair) in section
            pair isa AbstractVector && length(pair) == 2 ||
                throw(ArgumentError("$label.$name must be a two-element range"))
            lo, hi = Float64(pair[1]), Float64(pair[2])
            isfinite(lo) && isfinite(hi) && lo < hi ||
                throw(ArgumentError("$label.$name must be finite with lower < upper"))
            result[String(name)] = [lo, hi]
        end
        result
    end
    scalar_bounds = bounds(raw["scalar_bounds"], "scalar_bounds")
    temporal_bounds = bounds(raw["temporal_bounds"], "temporal_bounds")
    objective = raw["objective"]
    objective isa AbstractDict || throw(ArgumentError("objective must be an object"))
    fraction = Float64(get(objective, "min_completion_fraction", 0.9))
    isfinite(fraction) && 0 <= fraction <= 1 ||
        throw(ArgumentError("objective.min_completion_fraction must be in [0,1]"))
    seed_path = _preflight_path(base, raw["seed_config"], "seed_config")
    seed = load_json(seed_path)
    seed isa AbstractDict || throw(ArgumentError("seed_config must contain an object"))
    seed_paths = Dict{String,String}()
    _walk_seed_paths!(seed_paths, seed, "", dirname(seed_path))
    paths = Dict{String,String}("config"=>config_path, "seed_config"=>seed_path)
    for (k, v) in seed_paths
        key = k == "population_path" ? "population" :
              (occursin("covimod", lowercase(k)) ? "covimod" :
               (occursin("immunity", lowercase(k)) ? "immunity_events" : "seed.$k"))
        paths[key] = v
    end
    if haskey(raw, "gt_dir")
        gt_dir = _preflight_path(base, raw["gt_dir"], "gt_dir")
        isdir(gt_dir) || throw(ArgumentError("gt_dir must be a directory"))
        paths["ground_truth"] = gt_dir
        gt_manifest = _validate_gt_dir(gt_dir)
    else
        gt_manifest = Dict{String,Any}()
    end
    for field in ("julia_bin", "project_dir", "advanced_cli")
        haskey(raw, field) || throw(ArgumentError("missing executable path: $field"))
        paths[field] = _preflight_path(base, raw[field], field)
    end
    monthly_days = Int(get(raw, "monthly_days", 30))
    monthly_days > 0 || throw(ArgumentError("monthly_days must be positive"))
    weights = get(raw, "age_population_weights", DEFAULT_AGE_POPULATION_WEIGHTS)
    all(isfinite(Float64(v)) && Float64(v) > 0 for v in values(weights)) ||
        throw(ArgumentError("age_population_weights must be finite and positive"))
    total = sum(Float64(v) for v in values(weights))
    isfinite(total) && total > 0 || throw(ArgumentError("age_population_weights must have positive mass"))
    for field in ("output_dir",)
        haskey(raw, field) || throw(ArgumentError("missing path: $field"))
        paths[field] = isabspath(String(raw[field])) ? String(raw[field]) :
                       normpath(joinpath(base, String(raw[field])))
    end
    Dict{String,Any}(
        "valid"=>true, "config_path"=>config_path, "config_directory"=>base,
        "cwd"=>pwd(), "paths"=>paths, "path_identities"=>Dict(k=>Dict("exists"=>ispath(v),
        "type"=>isdir(v) ? "directory" : "file", "sha256"=>isfile(v) ? _path_hash(v) : "") for (k,v) in paths if ispath(v)),
        "stages"=>dimensions, "bounds"=>Dict("scalar"=>scalar_bounds, "temporal"=>temporal_bounds),
        "effective_completion_threshold"=>0.9, "source_completion_threshold"=>fraction,
        "ground_truth"=>gt_manifest, "invocation"=>Dict("advanced_cli"=>false, "slurm"=>false,
        "readiness"=>readiness), "output_root_created"=>false,
    )
end

function create_candidate_root(parent::String, candidate_id::String)
    mkpath(parent)
    base = joinpath(parent, candidate_id)
    root = base
    suffix = 0
    while ispath(root)
        suffix += 1
        root = "$base-$suffix"
    end
    mkpath(root)
    return root
end

function adapter_failure(class::String; command=String[], exit_code=nothing, timeout_seconds=nothing,
                         detail="")
    class in ADAPTER_FAILURE_CLASSES || throw(ArgumentError("unknown adapter failure class: $class"))
    Dict{String,Any}("status"=>"failed", "failure_class"=>class, "penalty"=>Inf,
        "command"=>String.(command), "exit_code"=>exit_code, "timeout_seconds"=>timeout_seconds,
        "diagnostic"=>String(detail), "ranking_eligible"=>false, "simulated"=>false)
end
