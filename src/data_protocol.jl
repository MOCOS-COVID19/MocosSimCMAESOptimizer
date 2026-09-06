const CANONICAL_GT_FILES = Dict(
    "daily_detections" => "daily_detections.csv",
    "daily_deaths" => "daily_deaths.csv",
    "daily_hospitalizations" => "daily_hospitalizations.csv",
    "daily_student_detections" => "daily_student_detections.csv",
)

function _expand_environment_variables(value::AbstractString, field::String)
    return replace(String(value), r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}" => matched -> begin
        name = matched[3:end-1]
        haskey(ENV, name) || throw(ArgumentError("$field requires environment variable $name"))
        ENV[name]
    end)
end

"""Create non-overlapping train, validation, and frozen-test day ranges."""
function temporal_data_split(total_days::Int; validation_days::Int=56,
                             test_days::Int=84, min_train_days::Int=1)
    total_days > 0 || throw(ArgumentError("total_days must be positive"))
    validation_days > 0 || throw(ArgumentError("validation_days must be positive"))
    test_days > 0 || throw(ArgumentError("test_days must be positive"))
    train_end = total_days - validation_days - test_days
    train_end >= min_train_days || throw(ArgumentError(
        "study period leaves fewer than $min_train_days training days"))
    validation_end = train_end + validation_days
    return Dict{String,Any}(
        "train" => Dict("start_day" => 1, "end_day" => train_end),
        "validation" => Dict("start_day" => train_end + 1, "end_day" => validation_end),
        "test" => Dict("start_day" => validation_end + 1, "end_day" => total_days,
                       "frozen" => true),
        "selection_max_day" => validation_end,
        "total_days" => total_days,
    )
end

"""Return the evaluation windows appropriate for one optimization stage.

Before the final test horizon, validation is the trailing rolling-origin window
of that stage. Once a stage reaches the frozen test, the predeclared final
train/validation/test boundaries are used unchanged.
"""
function stage_data_split(stage_days::Int, validation::AbstractDict)
    stage_days > 1 || throw(ArgumentError("a stage needs at least two days"))
    if !haskey(validation, "stage_validation_days") &&
       !haskey(validation, "test_start_day")
        holdout = min(max(Int(get(validation, "holdout_days", 28)), 1), stage_days)
        return Dict{String,Any}(
            "mode" => "legacy_diagnostic_holdout", "stage_days" => stage_days,
            "train" => Dict("start_day" => 1, "end_day" => stage_days),
            "validation" => Dict("start_day" => stage_days - holdout + 1,
                                 "end_day" => stage_days),
            "test" => nothing)
    end
    test_start = Int(get(validation, "test_start_day", typemax(Int)))
    if stage_days >= test_start
        train_end = Int(get(validation, "train_end_day", test_start - 1))
        validation_start = Int(get(validation, "validation_start_day", train_end + 1))
        validation_end = min(stage_days,
            Int(get(validation, "validation_end_day", test_start - 1)))
        train_end < validation_start <= validation_end < test_start ||
            throw(ArgumentError("final train/validation/test windows overlap or are empty"))
        return Dict{String,Any}(
            "mode" => "frozen_test", "stage_days" => stage_days,
            "train" => Dict("start_day" => 1, "end_day" => train_end),
            "validation" => Dict("start_day" => validation_start,
                                 "end_day" => validation_end),
            "test" => Dict("start_day" => test_start,
                           "end_day" => min(stage_days,
                               Int(get(validation, "test_end_day", stage_days))),
                           "frozen" => true))
    end
    requested = Int(get(validation, "stage_validation_days",
                        get(validation, "holdout_days", 28)))
    validation_days = min(max(requested, 1), stage_days - 1)
    train_end = stage_days - validation_days
    return Dict{String,Any}(
        "mode" => "rolling_origin", "stage_days" => stage_days,
        "train" => Dict("start_day" => 1, "end_day" => train_end),
        "validation" => Dict("start_day" => train_end + 1,
                             "end_day" => stage_days),
        "test" => nothing)
end

function _canonical_csv_rows(path::String)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("$(basename(path)) is empty"))
    header = lowercase.(strip.(split(lines[1], ',')))
    day_column = findfirst(==("day"), header)
    value_column = findfirst(x -> x in ("value", "observed", "observations",
                                        "7t_hospitalisierung_faelle"), header)
    day_column === nothing && throw(ArgumentError("$(basename(path)) has no day column"))
    value_column === nothing && throw(ArgumentError("$(basename(path)) has no value column"))
    rows = Tuple{Int,Float64}[]
    for (line_number, line) in enumerate(lines[2:end])
        columns = split(line, ',')
        max(day_column, value_column) <= length(columns) ||
            throw(ArgumentError("$(basename(path)) has a malformed row $(line_number + 1)"))
        day = tryparse(Int, strip(columns[day_column]))
        value = tryparse(Float64, strip(columns[value_column]))
        day === nothing && throw(ArgumentError("$(basename(path)) has an invalid day"))
        value === nothing && throw(ArgumentError("$(basename(path)) has an invalid value"))
        push!(rows, (day, value))
    end
    return rows
end

"""Build canonical observations and a fail-closed data-quality report.

Every canonical row contains `date/day/metric/value/source/status`. Required
metrics determine the common study cutoff; sparse auxiliary metrics do not.
"""
function canonical_data_protocol(gt_dir::String, start_date::Date;
                                 required_metrics::Vector{String}=[
                                     "daily_detections", "daily_deaths",
                                     "daily_hospitalizations"],
                                 study_end_day::Union{Nothing,Int}=nothing)
    observations = Dict{String,Any}[]
    metric_reports = Dict{String,Any}()
    required_end_days = Int[]
    for (metric, filename) in sort!(collect(CANONICAL_GT_FILES), by=first)
        path = joinpath(gt_dir, filename)
        required = metric in required_metrics
        if !isfile(path)
            required && throw(ArgumentError("required metric is absent: $filename"))
            metric_reports[metric] = Dict("status" => "absent", "required" => false)
            continue
        end
        rows = _canonical_csv_rows(path)
        isempty(rows) && throw(ArgumentError("$filename contains no observations"))
        days = first.(rows)
        values = last.(rows)
        duplicates = sort!([day for day in unique(days) if count(==(day), days) > 1])
        nonfinite = [day for ((day, value)) in rows if !isfinite(value)]
        # The supplied RKI-derived series uses -1 as an explicit unavailable
        # sentinel. Other negative counts are invalid.
        negative = [day for ((day, value)) in rows if value < 0 && value != -1]
        gaps = setdiff(collect(minimum(days):maximum(days)), unique(days))
        isempty(duplicates) || throw(ArgumentError("$filename has duplicate days: $(join(duplicates, ','))"))
        isempty(nonfinite) || throw(ArgumentError("$filename has non-finite values"))
        isempty(negative) || throw(ArgumentError("$filename has negative counts"))
        issorted(days) || throw(ArgumentError("$filename days are not increasing"))
        required && push!(required_end_days, maximum(days))
        metric_reports[metric] = Dict(
            "status" => "valid", "required" => required, "source" => filename,
            "observations" => length(rows), "start_day" => minimum(days),
            "end_day" => maximum(days), "gap_count" => length(gaps), "gaps" => gaps,
            "sha256" => bytes2hex(open(sha256, path)),
        )
        for (day, value) in rows
            push!(observations, Dict{String,Any}(
                "date" => string(start_date + Day(day - 1)), "day" => day,
                "metric" => metric, "value" => value == -1 ? nothing : value,
                "source" => filename,
                "status" => value == -1 ? "missing_sentinel" : "observed"))
        end
    end
    isempty(required_end_days) && throw(ArgumentError("no required metrics were loaded"))
    common_end = minimum(required_end_days)
    final_day = study_end_day === nothing ? common_end : min(study_end_day, common_end)
    final_day > 0 || throw(ArgumentError("study cutoff must be positive"))
    filter!(row -> Int(row["day"]) <= final_day, observations)
    sort!(observations, by=row -> (Int(row["day"]), String(row["metric"])))
    student_alias = joinpath(gt_dir, "daily_students_detections.csv")
    aliases = Dict{String,Any}()
    if isfile(student_alias) && isfile(joinpath(gt_dir, "daily_student_detections.csv"))
        canonical_student = joinpath(gt_dir, "daily_student_detections.csv")
        same_content = bytes2hex(open(sha256, student_alias)) ==
                       bytes2hex(open(sha256, canonical_student))
        same_content || throw(ArgumentError(
            "student detection alias differs from daily_student_detections.csv"))
        aliases["daily_students_detections.csv"] = Dict(
            "canonical" => "daily_student_detections.csv", "identical" => true)
    end
    return Dict{String,Any}(
        "schema_version" => 1, "day_one" => string(start_date),
        "timezone" => "Europe/Berlin", "common_required_end_day" => common_end,
        "study_end_day" => final_day, "metrics" => metric_reports,
        "observations" => observations, "aliases" => aliases,
    )
end

"""Negative-Binomial log likelihood using mean/dispersion parameterization."""
function negative_binomial_loglikelihood(observed::Real, predicted::Real,
                                         dispersion::Real)
    y = Float64(observed)
    mu = Float64(predicted)
    r = Float64(dispersion)
    isfinite(y) && y >= 0 || throw(ArgumentError("observed count must be nonnegative and finite"))
    isfinite(mu) && mu >= 0 || throw(ArgumentError("predicted mean must be nonnegative and finite"))
    isfinite(r) && r > 0 || throw(ArgumentError("dispersion must be positive and finite"))
    yi = round(Int, y)
    isapprox(y, yi; atol=1e-8) || throw(ArgumentError("observed count must be integral"))
    mu == 0 && return yi == 0 ? 0.0 : -Inf
    log_coefficient = 0.0
    for k in 1:yi
        log_coefficient += log(r + k - 1) - log(k)
    end
    return log_coefficient + r * log(r / (r + mu)) + yi * log(mu / (r + mu))
end
