push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MocosSimCMAESOptimizer

config_path = joinpath(@__DIR__, "..", "optimizer_config.json")
result = MocosSimCMAESOptimizer.run_optimizer(config_path)
println(result)
