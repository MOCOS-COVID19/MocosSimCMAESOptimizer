"""Versioned, source-backed policy calendar used by event parameterization."""
struct EventCalendar
    path::String
    content_hash::String
    start_date::Date
    end_date::Date
    events::Vector{Dict{String,Any}}
end

const EVENT_PARAMETER_CATEGORIES = Dict(
    "infection_modulation.params.interval_values" => "general_contacts",
    "school_contact_modulation.params.interval_values" => "school_contacts",
    "mild_detection_modulation.params.interval_values" => "detection",
    "tracing_modulation.params.interval_values" => "contact_tracing",
)
const EVENT_CATEGORIES = Set(values(EVENT_PARAMETER_CATEGORIES))

function validate_event_calendar(raw::AbstractDict, path::AbstractString)
    get(raw, "schema_version", nothing) == "saxony-policy-events-v1" ||
        throw(ArgumentError("event calendar has unsupported schema_version"))
    start_date = try
        Date(String(raw["simulation_start_date"]))
    catch
        throw(ArgumentError("event calendar simulation_start_date is invalid"))
    end
    end_date = try
        Date(String(raw["simulation_end_date"]))
    catch
        throw(ArgumentError("event calendar simulation_end_date is invalid"))
    end
    end_date >= start_date || throw(ArgumentError("event calendar date range is reversed"))
    events = get(raw, "events", nothing)
    events isa AbstractVector || throw(ArgumentError("event calendar events must be an array"))
    ids = Set{String}(); previous = nothing
    normalized = Dict{String,Any}[]
    required = ("id", "publication_date", "effective_date", "end_date", "geography",
                "conditional_status", "parameter_categories", "description", "source_url", "cited_range")
    for (index, item) in enumerate(events)
        item isa AbstractDict || throw(ArgumentError("event calendar entry $index is not an object"))
        all(k -> haskey(item, k), required) || throw(ArgumentError("event calendar entry $index misses required fields"))
        event = Dict{String,Any}(String(k) => v for (k,v) in item)
        id = String(event["id"]); isempty(id) && throw(ArgumentError("event id cannot be empty"))
        id in ids && throw(ArgumentError("duplicate event id: $id")); push!(ids, id)
        publication = try Date(String(event["publication_date"])) catch; throw(ArgumentError("invalid publication date for $id")); end
        effective = try Date(String(event["effective_date"])) catch; throw(ArgumentError("invalid effective date for $id")); end
        finish = try Date(String(event["end_date"])) catch; throw(ArgumentError("invalid end date for $id")); end
        start_date <= effective <= end_date || throw(ArgumentError("event $id is outside calendar range"))
        effective <= finish <= end_date || throw(ArgumentError("event $id has an invalid end date"))
        publication <= effective || throw(ArgumentError("event $id was published after it took effect"))
        previous === nothing || (effective, id) >= previous || throw(ArgumentError("events are not ordered by effective_date and id"))
        previous = (effective, id)
        String(event["geography"]) == "Saxony" || throw(ArgumentError("event $id has unsupported geography"))
        status = String(event["conditional_status"])
        status in ("unconditional", "conditional", "inactive") || throw(ArgumentError("event $id has invalid conditional_status"))
        cats = event["parameter_categories"]
        cats isa AbstractVector && !isempty(cats) || throw(ArgumentError("event $id has no parameter categories"))
        all(c -> String(c) in EVENT_CATEGORIES, cats) || throw(ArgumentError("event $id has an unknown parameter category"))
        url = String(event["source_url"]); startswith(url, "https://") || throw(ArgumentError("event $id source_url must use HTTPS"))
        isempty(strip(String(event["cited_range"]))) && throw(ArgumentError("event $id has no cited document range"))
        isempty(strip(String(event["description"]))) && throw(ArgumentError("event $id has no description"))
        push!(normalized, event)
    end
    return start_date, end_date, normalized
end

function load_event_calendar(path::AbstractString)
    absolute = abspath(path)
    bytes = read(absolute)
    raw = JSON.parse(String(bytes))
    start_date, end_date, events = validate_event_calendar(raw, absolute)
    return EventCalendar(absolute, bytes2hex(SHA.sha256(bytes)), start_date, end_date, events)
end

event_category_for_parameter(name::AbstractString) = get(EVENT_PARAMETER_CATEGORIES, String(name), nothing)

function event_change_points(calendar::EventCalendar, category::AbstractString; include_conditional::Bool=true)
    category in EVENT_CATEGORIES || throw(ArgumentError("unknown event category: $category"))
    points = Dict{Int,Vector{String}}(1 => String[])
    for event in calendar.events
        category in String.(event["parameter_categories"]) || continue
        status = String(event["conditional_status"])
        status == "inactive" && continue
        status == "conditional" && !include_conditional && continue
        start_day = Dates.value(Date(String(event["effective_date"])) - calendar.start_date) + 1
        end_day = Dates.value(Date(String(event["end_date"])) - calendar.start_date) + 2
        push!(get!(points, start_day, String[]), String(event["id"]))
        end_day <= Dates.value(calendar.end_date-calendar.start_date)+1 &&
            push!(get!(points, end_day, String[]), String(event["id"]))
    end
    return [(day=day, event_ids=sort(unique(ids))) for (day,ids) in sort(collect(points), by=first)]
end

function event_calendar_metadata(calendar::EventCalendar; horizon_days::Union{Nothing,Int}=nothing)
    ids = String[]
    for event in calendar.events
        day = Dates.value(Date(String(event["effective_date"])) - calendar.start_date) + 1
        (horizon_days === nothing || day <= horizon_days) && push!(ids, String(event["id"]))
    end
    return Dict{String,Any}("schema_version"=>"event-calendar-provenance-v1",
        "calendar_path"=>calendar.path, "calendar_sha256"=>calendar.content_hash,
        "event_ids"=>ids, "simulation_start_date"=>string(calendar.start_date))
end

function assert_event_calendar_resume!(artifact::AbstractDict, calendar::EventCalendar;
                                       seasonality::Union{Nothing,AbstractDict}=nothing)
    saved = get(artifact, "event_calendar", nothing)
    saved isa AbstractDict || throw(ArgumentError("resume refused: saved run has no event calendar provenance"))
    get(saved, "calendar_sha256", nothing) == calendar.content_hash ||
        throw(ArgumentError("resume refused: event calendar content hash differs from saved run"))
    if seasonality !== nothing
        get(saved, "seasonality", Dict{String,Any}()) == seasonality ||
            throw(ArgumentError("resume refused: event seasonality differs from saved run"))
    end
    return true
end

const EVENT_SEASONAL_TARGET = "infection_modulation.params.interval_values"

"""Validate the deterministic seasonal layer for out-of-household infectivity."""
function validate_event_seasonality(config::AbstractDict)
    isempty(config) && return true
    Bool(get(config, "enabled", false)) || return true
    String(get(config, "target_parameter", EVENT_SEASONAL_TARGET)) == EVENT_SEASONAL_TARGET ||
        throw(ArgumentError("event seasonality may only target out-of-household infection modulation"))
    String(get(config, "applies_to", "out_of_household_contacts")) == "out_of_household_contacts" ||
        throw(ArgumentError("event seasonality applies_to must be out_of_household_contacts"))
    reduction = Float64(get(config, "summer_reduction", 0.0))
    0.0 <= reduction < 1.0 || throw(ArgumentError("event_seasonality.summer_reduction must be in [0,1)"))
    String(get(config, "shape", "annual_cosine")) == "annual_cosine" ||
        throw(ArgumentError("event_seasonality.shape must be annual_cosine"))
    peak = String(get(config, "summer_peak", "07-15"))
    occursin(r"^\d\d-\d\d$", peak) || throw(ArgumentError("event_seasonality.summer_peak must use MM-DD"))
    try
        Date("2000-$peak")
    catch
        throw(ArgumentError("event_seasonality.summer_peak is not a valid month and day"))
    end
    return true
end

"""Annual cosine multiplier: 1-reduction at the summer peak and 1 in winter."""
function seasonal_out_of_household_multiplier(calendar::EventCalendar, simulation_day::Real,
                                                config::AbstractDict)
    validate_event_seasonality(config)
    (!Bool(get(config, "enabled", false)) || isempty(config)) && return 1.0
    day = max(Int(ceil(Float64(simulation_day))), 1)
    date = calendar.start_date + Day(day - 1)
    month, dom = parse.(Int, split(String(get(config, "summer_peak", "07-15")), '-'))
    peak = Date(year(date), month, dom)
    # Wrap to the nearest occurrence, which keeps the curve continuous across years.
    offsets = [Dates.value(date - (peak + Year(delta))) for delta in -1:1]
    distance = offsets[argmin(abs.(offsets))]
    reduction = Float64(get(config, "summer_reduction", 0.0))
    return 1.0 - reduction * (1.0 + cos(2pi * distance / 365.2425)) / 2.0
end

"""Apply seasonality only to the simulator's out-of-household infectivity path."""
function apply_event_seasonality!(candidate::Dict{String,Any}, calendar::EventCalendar,
                                  config::AbstractDict, horizon_days::Int)
    isempty(config) && return candidate
    validate_event_seasonality(config)
    Bool(get(config, "enabled", false)) || return candidate
    values = get_nested(candidate, EVENT_SEASONAL_TARGET)
    times = get_nested(candidate, replace(EVENT_SEASONAL_TARGET, "interval_values"=>"interval_times"))
    for i in eachindex(values)
        day = Float64(times[min(i, length(times))])
        day <= horizon_days || continue
        values[i] = Float64(values[i]) * seasonal_out_of_household_multiplier(calendar, day, config)
    end
    set_nested!(candidate, EVENT_SEASONAL_TARGET, values)
    return candidate
end
