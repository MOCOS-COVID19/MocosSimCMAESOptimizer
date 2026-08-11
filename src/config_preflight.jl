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
            lines = collect(eachline(io))
            isempty(lines) && throw(ArgumentError("ground_truth.$file has no header"))
            header = lowercase.(strip.(split(lines[1], ',')))
            length(header) >= 2 && ("day" in header) && any(x -> x in ("value", "observed", "observations"), header) ||
                throw(ArgumentError("ground_truth.$file has invalid day/value header"))
            for (line_number, line) in zip(2:length(lines), lines[2:end])
                parts = split(line, ',')
                length(parts) >= 2 || throw(ArgumentError("ground_truth.$file malformed row $line_number"))
                day = try parse(Int, strip(parts[1])) catch
                    throw(ArgumentError("ground_truth.$file invalid day at row $line_number"))
                end
                value = try parse(Float64, strip(parts[2])) catch
                    throw(ArgumentError("ground_truth.$file invalid value at row $line_number"))
                end
                isfinite(value) || throw(ArgumentError("ground_truth.$file non-finite value at row $line_number"))
                push!(rows, (day, value))
            end
        end
        days = first.(rows)
        length(unique(days)) == length(days) || throw(ArgumentError("ground_truth.$file has duplicate day labels"))
        all(>(0), days) || throw(ArgumentError("ground_truth.$file has nonpositive day labels"))
        issorted(days) || throw(ArgumentError("ground_truth.$file day labels must be strictly increasing"))
        isempty(rows) && throw(ArgumentError("ground_truth.$file has no parseable observations"))
        csvs[file] = Dict("path"=>path, "observations"=>length(rows),
                          "days"=>days, "valid"=>true,
                          "day_policy"=>"positive_unique_strictly_increasing")
    end
    isempty(csvs) && throw(ArgumentError("ground_truth has no CSV files"))
    known_optional = ["daily_student_detections", "household_infections",
                      "household_infection_rate"]
    merge!(csvs, Dict{String,Any}(
        "optional_fields" => Dict(name => Dict("present" => false, "status" => "absent")
                                  for name in known_optional)))
    return csvs
end

function _seed_nested(seed::AbstractDict, path::String)
    node = seed
    for part in split(path, '.')
        node isa AbstractDict && haskey(node, part) ||
            throw(ArgumentError("seed missing required key: $path"))
        node = node[part]
    end
    return node
end

function _validate_jld2_schema(path::String, kind::Symbol)
    isfile(path) || throw(ArgumentError("$kind input does not exist: $path"))
    try
        HDF5.h5open(path, "r") do file
            keys_present = Set(String.(collect(keys(file))))
            if kind == :population
                "individuals_df" in keys_present ||
                    throw(ArgumentError("population missing required key individuals_df"))
                data = file["individuals_df"]
                dims = collect(size(data))
                isempty(dims) || prod(dims) > 0 ||
                    throw(ArgumentError("population individuals_df is empty"))
                return Dict{String,Any}("path" => path, "required_keys" => ["individuals_df"],
                    "dimensions" => dims, "rows" => isempty(dims) ? 0 : dims[1])
            elseif kind == :covimod
                required = ["age_thresholds", "contact_mat", "uses_genders"]
                all(x -> x in keys_present, required) ||
                    throw(ArgumentError("covimod missing required key(s): " *
                        join(filter(x -> !(x in keys_present), required), ", ")))
                thresholds = read(file["age_thresholds"])
                matrix = file["contact_mat"]
                mdims = collect(size(matrix))
                length(mdims) == 2 && mdims[1] == mdims[2] &&
                    mdims[1] == length(thresholds) ||
                    throw(ArgumentError("covimod contact_mat dimensions incompatible with age_thresholds"))
                all(isfinite, Float64.(thresholds)) ||
                    throw(ArgumentError("covimod age_thresholds contains non-finite values"))
                return Dict{String,Any}("path" => path, "required_keys" => required,
                    "age_threshold_count" => length(thresholds),
                    "contact_matrix_shape" => mdims,
                    "uses_genders" => Bool(read(file["uses_genders"])))
            elseif kind == :immunity_events
                "events" in keys_present ||
                    throw(ArgumentError("immunity_events missing required key events"))
                dims = collect(size(file["events"]))
                isempty(dims) || prod(dims) > 0 ||
                    throw(ArgumentError("immunity_events events is empty"))
                return Dict{String,Any}("path" => path, "required_keys" => ["events"],
                    "dimensions" => dims, "event_count" => isempty(dims) ? 1 : prod(dims))
            end
            throw(ArgumentError("unknown model input kind: $kind"))
        end
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("$kind input is unreadable or malformed: $path ($(sprint(showerror, err)))"))
    end
end

function _validate_seed_model_inputs(seed::AbstractDict, seed_path::String,
                                     stage_months::Int, monthly_days::Int,
                                     scalar_bounds::AbstractDict,
                                     temporal_bounds::AbstractDict)
    population_value = _seed_nested(seed, "population_path")
    population_value isa AbstractString ||
        throw(ArgumentError("seed.population_path must be a path string"))
    covimod_value = _seed_nested(seed, "transmission_probabilities.age_coupling_data_path")
    covimod_value isa AbstractString ||
        throw(ArgumentError("seed.transmission_probabilities.age_coupling_data_path must be a path string"))
    immunity_value = if haskey(seed, "immunity_events_path")
        seed["immunity_events_path"]
    else
        _seed_nested(seed, "initial_conditions.immunization.immunity_events")
    end
    immunity_value isa AbstractString ||
        throw(ArgumentError("seed.initial_conditions.immunization.immunity_events must be a path string"))
    resolve(value) = isabspath(String(value)) ? String(value) :
        normpath(joinpath(dirname(seed_path), String(value)))
    population_path, covimod_path, immunity_path =
        resolve(population_value), resolve(covimod_value), resolve(immunity_value)
    model = Dict{String,Any}(
        "population" => _validate_jld2_schema(population_path, :population),
        "covimod" => _validate_jld2_schema(covimod_path, :covimod),
        "immunity_events" => _validate_jld2_schema(immunity_path, :immunity_events))
    for name in keys(scalar_bounds)
        value = _seed_nested(seed, String(name))
        value isa Number && isfinite(Float64(value)) ||
            throw(ArgumentError("seed.$name must be a finite numeric scalar"))
    end
    required_days = stage_months * monthly_days
    for (name, pair) in temporal_bounds
        path = String(name)
        values = _seed_nested(seed, path)
        values isa AbstractVector || throw(ArgumentError("seed.$path must be a vector"))
        times_path = replace(path, r"\.interval_values$" => ".interval_times")
        times = _seed_nested(seed, times_path)
        times isa AbstractVector && length(times) == length(values) ||
            throw(ArgumentError("seed.$path and $times_path dimensions disagree"))
        isempty(times) && throw(ArgumentError("seed.$path interval schema is empty"))
        all(x -> x isa Number && isfinite(Float64(x)) && Float64(x) > 0, times) ||
            throw(ArgumentError("seed.$times_path must contain positive finite days"))
        all(x -> x isa Number && isfinite(Float64(x)), values) ||
            throw(ArgumentError("seed.$path must contain finite numeric values"))
        all(diff(Float64.(times)) .> 0) ||
            throw(ArgumentError("seed.$times_path must be strictly increasing"))
        maximum(Float64.(times)) >= required_days ||
            throw(ArgumentError("seed.$times_path does not cover requested horizon $required_days days"))
    end
    return model
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
    isempty(stages) && throw(ArgumentError("stages must not be empty"))
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
    monthly_days = Int(get(raw, "monthly_days", 30))
    monthly_days > 0 || throw(ArgumentError("monthly_days must be positive"))
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
    model_inputs = _validate_seed_model_inputs(seed, seed_path, first(stages)["fit_months"],
                                               monthly_days, scalar_bounds, temporal_bounds)
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
        "model_inputs"=>model_inputs, "ground_truth"=>gt_manifest, "invocation"=>Dict("advanced_cli"=>false, "slurm"=>false,
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
