using Test
using JSON
include(joinpath(@__DIR__, "..", "src", "MocosSimCMAESOptimizer.jl"))
const O = MocosSimCMAESOptimizer

const CALENDAR_PATH = joinpath(@__DIR__, "..", "data", "saxony_policy_events_2020-09-03_2022-08-31.json")

@testset "event calendar schema, dates, categories and conditions" begin
    calendar = O.load_event_calendar(CALENDAR_PATH)
    @test calendar.start_date == O.Date("2020-09-03")
    general = O.event_change_points(calendar, "general_contacts")
    @test any(p -> p.day == 61 && "SN-2020-11-02-CONTACTS" in p.event_ids, general)
    with_conditional = O.event_change_points(calendar, "detection")
    without_conditional = O.event_change_points(calendar, "detection"; include_conditional=false)
    @test length(with_conditional) > length(without_conditional)
    @test O.event_category_for_parameter("mild_detection_modulation.params.interval_values") == "detection"
    @test O.event_category_for_parameter("tracing_modulation.params.interval_values") == "contact_tracing"
end

@testset "event buckets map to simulator days and parameters" begin
    calendar = O.load_event_calendar(CALENDAR_PATH)
    seed = Dict{String,Any}(
        "stop_simulation_time"=>900,
        "infection_modulation"=>Dict("params"=>Dict("interval_times"=>[1,60,61,90,120,180,365], "interval_values"=>fill(0.5,7))),
        "mild_detection_modulation"=>Dict("params"=>Dict("interval_times"=>[1,100,187,250,365], "interval_values"=>fill(0.4,5))),
        "tracing_modulation"=>Dict("params"=>Dict("interval_times"=>[1,61,100,365], "interval_values"=>fill(0.3,4))))
    objective = O.ObjectiveConfig(Dict{String,Float64}(),1,1.0,0,"baseline",0.0,0.0)
    posterior = O.PosteriorConfig(false,"none",1,1,1,0.1,1.0,1.0,1.0,1.0,0.0)
    bounds = Dict(k=>(0.0,1.0) for k in ["infection_modulation.params.interval_values", "mild_detection_modulation.params.interval_values", "tracing_modulation.params.interval_values"])
    cfg = O.OptimizerConfig("seed","out",30,O.StageConfig[],Dict{String,Tuple{Float64,Float64}}(),bounds,
        Dict{String,Dict{String,Any}}(),"events",Dict("all"=>1.0),Dict{String,Any}(),objective,nothing,
        Dict{String,Vector{String}}(),nothing,posterior,seed,calendar)
    O.CURRENT_OPTIMIZER_CONFIG[] = cfg
    specs = O.build_specs(seed,cfg)
    infection = only(filter(s -> s.name == "infection_modulation.params.interval_values", specs))
    @test infection.length == length(O.event_change_points(calendar,"general_contacts"))
    @test O.temporal_active_length(seed,infection,3,cfg) == count(p -> p.day <= 90, O.event_change_points(calendar,"general_contacts"))
    ranges = O.temporal_bucket_day_ranges(seed,infection,3,cfg)
    @test first(ranges) == (1,60)
    x = O.initial_vector(seed,specs); fill!(x,0.8)
    candidate = O.vector_to_config(seed,specs,x,3)
    @test candidate["stop_simulation_time"] == 90
    @test candidate["infection_modulation"]["params"]["interval_values"][3] == 0.8
end

@testset "resume refuses a changed calendar" begin
    calendar = O.load_event_calendar(CALENDAR_PATH)
    artifact = Dict("event_calendar"=>O.event_calendar_metadata(calendar))
    @test O.assert_event_calendar_resume!(artifact,calendar)
    artifact["event_calendar"]["calendar_sha256"] = "changed"
    @test_throws ArgumentError O.assert_event_calendar_resume!(artifact,calendar)
    artifact["event_calendar"] = O.event_calendar_metadata(calendar)
    artifact["event_calendar"]["seasonality"] = Dict("enabled"=>true, "summer_reduction"=>0.3)
    @test_throws ArgumentError O.assert_event_calendar_resume!(artifact, calendar;
        seasonality=Dict("enabled"=>true, "summer_reduction"=>0.2))
end

@testset "seasonality reduces only summer out-of-household infectivity" begin
    calendar = O.load_event_calendar(CALENDAR_PATH)
    seasonal = Dict{String,Any}(
        "enabled"=>true, "applies_to"=>"out_of_household_contacts",
        "target_parameter"=>"infection_modulation.params.interval_values",
        "summer_reduction"=>0.3, "summer_peak"=>"07-15")
    summer_day = O.Dates.value(O.Date("2021-07-15") - calendar.start_date) + 1
    winter_day = O.Dates.value(O.Date("2021-01-15") - calendar.start_date) + 1
    @test O.seasonal_out_of_household_multiplier(calendar, summer_day, seasonal) ≈ 0.7
    @test O.seasonal_out_of_household_multiplier(calendar, winter_day, seasonal) > 0.99

    candidate = Dict{String,Any}(
        "infection_modulation"=>Dict("params"=>Dict(
            "interval_times"=>[winter_day, summer_day], "interval_values"=>[0.8,0.8])),
        "transmission_probabilities"=>Dict("household"=>0.42))
    O.apply_event_seasonality!(candidate, calendar, seasonal, summer_day)
    @test candidate["infection_modulation"]["params"]["interval_values"][2] ≈ 0.56
    @test candidate["infection_modulation"]["params"]["interval_values"][1] > 0.79
    @test candidate["transmission_probabilities"]["household"] == 0.42
    @test_throws ArgumentError O.validate_event_seasonality(Dict(
        "enabled"=>true, "summer_reduction"=>1.2))
end
