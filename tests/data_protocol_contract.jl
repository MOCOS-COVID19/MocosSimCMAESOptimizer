using Test
using Dates

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

const O = MocosSimCMAESOptimizer

@testset "temporal split keeps frozen test outside model selection" begin
    split = O.temporal_data_split(900; validation_days=56, test_days=84)
    @test split["train"] == Dict("start_day" => 1, "end_day" => 760)
    @test split["validation"] == Dict("start_day" => 761, "end_day" => 816)
    @test split["test"]["start_day"] == 817
    @test split["test"]["end_day"] == 900
    @test split["test"]["frozen"]
    @test split["selection_max_day"] < split["test"]["start_day"]
    @test_throws ArgumentError O.temporal_data_split(100; validation_days=56, test_days=84)
end

@testset "each calibration stage gets its own rolling validation" begin
    validation = Dict{String,Any}(
        "stage_validation_days" => 28,
        "train_end_day" => 760, "validation_start_day" => 761,
        "validation_end_day" => 816, "test_start_day" => 817,
        "test_end_day" => 900)
    for (days, train_end) in ((90, 62), (180, 152), (360, 332),
                              (540, 512), (720, 692))
        split = O.stage_data_split(days, validation)
        @test split["mode"] == "rolling_origin"
        @test split["train"]["end_day"] == train_end
        @test split["validation"]["end_day"] == days
        @test split["test"] === nothing
    end
    final = O.stage_data_split(900, validation)
    @test final["mode"] == "frozen_test"
    @test final["train"]["end_day"] == 760
    @test final["validation"] == Dict("start_day" => 761, "end_day" => 816)
    @test final["test"]["start_day"] == 817
end

@testset "canonical calendar and data quality" begin
    root = mktempdir()
    for metric in ("daily_detections", "daily_deaths", "daily_hospitalizations")
        write(joinpath(root, "$metric.csv"), "day,value\n1,2\n2,0\n4,3\n")
    end
    report = O.canonical_data_protocol(root, Date(2020, 9, 3))
    @test report["common_required_end_day"] == 4
    @test report["metrics"]["daily_detections"]["gaps"] == [3]
    @test report["observations"][1]["date"] == "2020-09-03"
    @test all(haskey(row, key) for row in report["observations"] for
              key in ("date", "day", "metric", "value", "source", "status"))

    write(joinpath(root, "daily_deaths.csv"), "day,value\n1,-2\n")
    @test_throws ArgumentError O.canonical_data_protocol(root, Date(2020, 9, 3))
end

@testset "negative-binomial observation model" begin
    exact = O.negative_binomial_loglikelihood(12, 12, 20)
    shifted = O.negative_binomial_loglikelihood(12, 40, 20)
    @test isfinite(exact)
    @test exact > shifted
    @test O.negative_binomial_loglikelihood(0, 0, 20) == 0.0
    @test_throws ArgumentError O.negative_binomial_loglikelihood(-1, 1, 20)
    @test_throws ArgumentError O.negative_binomial_loglikelihood(1.5, 1, 20)
end
