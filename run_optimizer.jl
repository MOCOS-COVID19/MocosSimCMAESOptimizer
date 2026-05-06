#!/usr/bin/env julia
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using MocosSimCMAESOptimizer
MocosSimCMAESOptimizer.main()
