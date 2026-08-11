using Test
using JSON
using HDF5
using SHA

const REPO = normpath(joinpath(@__DIR__, ".."))
const JULIA = get(ENV, "JULIA", "/Users/marcinbodych/Workspace/saxocov/julia-1.7.0/bin/julia")
const DEFAULT_FIXTURE_SUBPROCESS_TIMEOUT_SECONDS = 30.0

"""Run a fixture subprocess without allowing a hung child to hang validation.

Both output streams are redirected to independent files so a chatty child
cannot block on pipe capacity; they are drained after termination. A timeout
is a blocked fixture, never a successful or failed pipeline result, and its
evidence is retained under a fresh fixture root.
"""
function run_fixture_subprocess(command::Cmd, phase::String, batch_path::String;
                                timeout_seconds::Float64 = try
                                    parse(Float64, get(ENV,
                                        "FIXTURE_SUBPROCESS_TIMEOUT_SECONDS", "30"))
                                catch
                                    DEFAULT_FIXTURE_SUBPROCESS_TIMEOUT_SECONDS
                                end,
                                poll_seconds::Float64 = 0.05)
    timeout_seconds > 0 || error("fixture subprocess timeout must be positive")
    started = time()
    batch_dir = dirname(abspath(batch_path))
    output_dir = mktempdir(batch_dir; prefix=".fixture-subprocess-")
    stdout_path = joinpath(output_dir, "stdout")
    stderr_path = joinpath(output_dir, "stderr")
    stdout_io, stderr_io = open(stdout_path, "w"), open(stderr_path, "w")
    process = run(pipeline(command, stdout_io, stderr_io); wait=false)
    close(stdout_io)
    close(stderr_io)
    timed_out = false
    while process_running(process)
        if time() - started >= timeout_seconds
            timed_out = true
            try
                kill(process, 15)
            catch
                # The process may have exited between polling and termination.
            end
            # Julia children can be inside a long sleep and defer TERM.  Force
            # termination after a short grace period so validation itself
            # remains bounded.
            sleep(0.1)
            if process_running(process)
                try
                    kill(process, 9)
                catch
                end
            end
            break
        end
        sleep(min(poll_seconds, max(timeout_seconds - (time() - started), 0.001)))
    end
    wait(process)
    stdout_text = read(stdout_path, String)
    stderr_text = read(stderr_path, String)
    elapsed = time() - started
    status = timed_out ? "blocked" : (success(process) ? "completed" : "failed")
    result = Dict{String,Any}(
        "status" => status,
        "phase" => phase,
        "command" => string(command),
        "timeout_seconds" => timeout_seconds,
        "elapsed_seconds" => elapsed,
        "exit_code" => try process.exitcode catch; nothing end,
        "stdout" => stdout_text,
        "stderr" => stderr_text,
    )
    if timed_out
        blocked_root = joinpath(batch_dir, "fixture_blocked")
        ispath(blocked_root) && error("fixture blocked evidence root already exists: $blocked_root")
        mkpath(blocked_root)
        result["evidence_root"] = blocked_root
        result["blocked_state_path"] = joinpath(blocked_root, "blocked_state.json")
        open(result["blocked_state_path"], "w") do io
            JSON.print(io, Dict(
                "status" => "fixture_blocked",
                "phase" => phase,
                "command" => string(command),
                "timeout_seconds" => timeout_seconds,
                "elapsed_seconds" => elapsed,
                "exit_code" => result["exit_code"],
                "stdout" => stdout_text,
                "stderr" => stderr_text,
                "batch_path" => abspath(batch_path),
            ))
        end
    else
        rm(output_dir; recursive=true, force=true)
    end
    return result
end

function require_fixture_subprocess(result::Dict{String,Any}, phase::String)
    @test result["status"] == "completed"
    result["status"] == "completed" ||
        error("fixture subprocess blocked or failed during $phase; evidence=$(get(result, "blocked_state_path", "none"))")
    return result
end

@testset "fixture subprocess timeout is bounded and durable" begin
    root = mktempdir()
    script = joinpath(root, "hang.jl")
    write(script, "println(\"fixture-started\"); flush(stdout); sleep(60)\n")
    batch_path = joinpath(root, "batch.json")
    write(batch_path, "{}")
    result = run_fixture_subprocess(
        `$JULIA --startup-file=no $script`, "timeout-regression", batch_path;
        timeout_seconds=2.0,
    )
    @test result["status"] == "blocked"
    @test result["elapsed_seconds"] < 5.0
    blocked_path = result["blocked_state_path"]
    @test isfile(blocked_path)
    blocked = JSON.parsefile(blocked_path)
    @test blocked["status"] == "fixture_blocked"
    @test blocked["phase"] == "timeout-regression"
    @test occursin("fixture-started", blocked["stdout"])
    @test blocked["command"] == result["command"]
end

@testset "fixture-only staged two-year pipeline" begin
    root = mktempdir()
    cfgdir = joinpath(root, "cfg")
    mkpath(cfgdir)
    h5open(joinpath(cfgdir, "population.jld2"), "w") do f
        write(f, "individuals_df", ones(Float64, 4, 2))
    end
    h5open(joinpath(cfgdir, "covimod.jld2"), "w") do f
        write(f, "age_thresholds", [0.0, 1.0, 2.0])
        write(f, "contact_mat", ones(Float64, 3, 3))
        write(f, "uses_genders", false)
    end
    h5open(joinpath(cfgdir, "events.jld2"), "w") do f
        write(f, "events", ones(Int, 2))
    end
    seed = Dict("population_path"=>"population.jld2",
        "transmission_probabilities"=>Dict("age_coupling_data_path"=>"covimod.jld2",
                                            "constant"=>0.1),
        "initial_conditions"=>Dict("immunization"=>Dict("immunity_events"=>"events.jld2")),
        "infection_modulation"=>Dict("function"=>"IntervalsModulations",
            "params"=>Dict("interval_times"=>[30, 60, 90, 120, 150, 180, 210, 240],
                           "interval_values"=>fill(0.2, 8))))
    open(joinpath(cfgdir, "seed.json"), "w") do io JSON.print(io, seed) end
    mkpath(joinpath(cfgdir, "gt"))
    write(joinpath(cfgdir, "gt", "daily.csv"), "day,value\n1,1\n2,2\n")
    cfg = Dict("seed_config"=>"seed.json", "output_dir"=>"unused",
        "monthly_days"=>30,
        "stages"=>[Dict("name"=>"fixture", "fit_months"=>1, "max_iterations"=>1,
                        "population_size"=>1, "sigma"=>0.1)],
        "scalar_bounds"=>Dict("transmission_probabilities.constant"=>[0.0, 1.0]),
        "temporal_bounds"=>Dict{String,Any}(),
        "objective"=>Dict("weights"=>Dict("daily_detections"=>1.0)),
        "validation"=>Dict("enabled"=>false), "age_population_weights"=>Dict("all"=>1.0),
        "gt_dir"=>"gt", "julia_bin"=>"fake-julia", "project_dir"=>".",
        "advanced_cli"=>"advanced_cli.jl")
    write(joinpath(cfgdir, "fake-julia"), "#!/bin/sh\nexit 0\n")
    write(joinpath(cfgdir, "advanced_cli.jl"), "# fixture boundary\n")
    chmod(joinpath(cfgdir, "fake-julia"), 0o755)
    config_path = joinpath(cfgdir, "config.json")
    open(config_path, "w") do io JSON.print(io, cfg) end
    batch = Dict("base_config"=>"config.json", "output_root"=>"fixture-run",
        "adapter_mode"=>"fixture", "target_months"=>24,
        "stages"=>[Dict("name"=>"stage_06m", "fit_months"=>6),
                   Dict("name"=>"stage_12m", "fit_months"=>12),
                   Dict("name"=>"stage_18m", "fit_months"=>18),
                   Dict("name"=>"stage_24m", "fit_months"=>24)])
    batch_path = joinpath(cfgdir, "batch.json")
    open(batch_path, "w") do io JSON.print(io, batch) end
    startup = run_fixture_subprocess(
        `$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl $batch_path`,
        "startup/preflight", batch_path; timeout_seconds=120.0)
    require_fixture_subprocess(startup, "startup/preflight")
    summary = JSON.parse(startup["stdout"])
    @test summary["status"] == "fixture_complete"
    @test summary["target_months"] == 24
    @test summary["deferred"]["simulation"] == "DEFERRED"
    @test summary["deferred"]["multi_seed"] == "DEFERRED"
    @test summary["deferred"]["slurm"] == "DEFERRED"
    @test length(summary["stages"]) == 4
    @test [s["fit_months"] for s in summary["stages"]] == [6, 12, 18, 24]
    @test all(s["gate"]["status"] == "passed" for s in summary["stages"])
    @test all(isfile(joinpath(s["stage_root"], "preflight_manifest.json"))
              for s in summary["stages"])
        @test all(isfile(joinpath(s["stage_root"], "metrics.json")) &&
                  isfile(joinpath(s["stage_root"], "survivor_selection_report.json")) &&
                  isfile(joinpath(s["stage_root"], "provenance_validation.json"))
                  for s in summary["stages"])
    # Trusted trajectory/CMA state must be durable at every handoff, not just
    # represented by the scalar archive identifiers.
    trajectory_ids = String[]
    for (i, stage) in enumerate(summary["stages"])
        state = JSON.parsefile(joinpath(stage["stage_root"], "stage_state.json"))
        reusable = JSON.parsefile(joinpath(stage["stage_root"], "full_reusable_state.json"))
        metrics = JSON.parsefile(joinpath(stage["stage_root"], "metrics.json"))
        selection = JSON.parsefile(joinpath(stage["stage_root"], "survivor_selection_report.json"))
        provenance = JSON.parsefile(joinpath(stage["stage_root"], "provenance_validation.json"))
        @test provenance["status"] == "consistent"
        @test isempty(provenance["contradictions"])
        @test selection["archive_count"] == length(JSON.parsefile(joinpath(stage["stage_root"], "survivor_archive.json")))
        @test selection["effective_quality_band"] == selection["quality_band"]
        @test selection["adaptive_target_status"] in ("met", "constrained")
        @test haskey(selection, "diversity")
        @test selection["rejected_total"] == sum(Int.(values(selection["rejected_counts"])))
        @test all(haskey(row, "source_config_identity") &&
                  haskey(row, "source_seed_identity") &&
                  haskey(row, "horizon") &&
                  haskey(row, "adapter_mode") &&
                  haskey(row, "metric_version") &&
                  haskey(row, "score_evidence") &&
                  haskey(row, "output_paths") &&
                  haskey(row, "failure_class")
                  for row in metrics)
        @test reusable["selection_report_path"] == joinpath(stage["stage_root"], "survivor_selection_report.json")
        @test reusable["selection_report_archive_ids"] == stage["archive_ids"]
        @test haskey(state, "trajectory_identity")
        @test haskey(state, "historical_trajectory")
        @test haskey(state, "prefix_hash")
        @test haskey(state, "locked_intervals")
        @test haskey(reusable, "cma_state")
        @test haskey(reusable, "trajectory_identity")
        @test reusable["admitted_ids"] == state["transfer_archive_ids"]
        push!(trajectory_ids, String(state["trajectory_identity"]))
        if i > 1
            previous = JSON.parsefile(joinpath(summary["stages"][i - 1]["stage_root"], "stage_state.json"))
            @test state["trajectory_identity"] == previous["trajectory_identity"]
            @test state["historical_trajectory"]["prefix_values"] ==
                  previous["historical_trajectory"]["values"]
            @test state["historical_trajectory"]["prefix_hash"] ==
                  bytes2hex(SHA.sha256(JSON.json(previous["historical_trajectory"]["values"])))
            @test all(get(c, "candidate_class", "") == "archive_transfer"
                      for c in JSON.parsefile(joinpath(stage["stage_root"], "transfer_candidates.json")))
        end
    end
    @test length(unique(trajectory_ids)) == 1
    # Every extension must consume the canonical archive finalized by its
    # immediate predecessor, while current-stage survivor selection remains a
    # separate artifact.
    for i in 2:length(summary["stages"])
        previous = summary["stages"][i - 1]
        current = summary["stages"][i]
        transfer = JSON.parsefile(joinpath(current["stage_root"], "transfer_manifest.json"))
        @test transfer["source_archive_path"] == previous["archive_path"]
        @test transfer["source_stage"] == previous["name"]
        @test transfer["target_stage"] == current["name"]
        @test transfer["source_horizon_months"] == previous["fit_months"]
        @test transfer["admitted_ids"] == previous["archive_ids"]
        @test transfer["candidate_order"] == previous["archive_ids"]
        @test transfer["protected_transfer_slots"] == previous["archive_ids"]
        @test isempty(intersect(Set(transfer["protected_transfer_slots"]),
                                Set(transfer["immigrant_slots"])))
        transferred = JSON.parsefile(joinpath(current["stage_root"], "transfer_candidates.json"))
        predecessor_archive = JSON.parsefile(joinpath(previous["stage_root"], "survivor_archive.json"))
        @test [c["candidate"] for c in transferred] == [c["candidate"] for c in predecessor_archive]
        @test [c["evaluated_vector"] for c in transferred] == [c["evaluated_vector"] for c in predecessor_archive]
        @test current["archive_ids"] != transfer["admitted_ids"]
    end
    @test isfile(joinpath(summary["output_root"], "pipeline_summary.json"))
    @test !isdir(joinpath(summary["output_root"], "advanced_cli.jl"))

    # An interrupted root may lose only its derived summary.  Resume must
    # reconstruct it from the committed, content-validated stage artifacts.
    summary_bytes = read(joinpath(summary["output_root"], "pipeline_summary.json"))
    rm(joinpath(summary["output_root"], "pipeline_summary.json"))
    resume = run_fixture_subprocess(
        `$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl $batch_path`,
        "resume", batch_path; timeout_seconds=120.0)
    require_fixture_subprocess(resume, "resume")
    reconstructed = JSON.parse(resume["stdout"])
    @test reconstructed == summary
    @test isfile(joinpath(summary["output_root"], "pipeline_summary.json"))

    # The canonical survivor archive and transfer manifest are independent
    # artifacts.  Changing either lineage path must fail closed.
    manifest_path = joinpath(summary["stages"][2]["stage_root"], "transfer_manifest.json")
    original_manifest = read(manifest_path)
    original_text = String(copy(original_manifest))
    tampered_manifest = JSON.parse(original_text)
    tampered_manifest["source_archive_path"] = joinpath(summary["output_root"], "wrong-survivor_archive.json")
    open(manifest_path, "w") do io JSON.print(io, tampered_manifest) end
    tampered_proc = run_fixture_subprocess(
        `$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl $batch_path`,
        "expected-rejection/lineage", batch_path; timeout_seconds=120.0)
    @test tampered_proc["status"] == "failed"
    @test tampered_proc["exit_code"] != 0
    open(manifest_path, "w") do io
        write(io, original_manifest)
    end
    @test JSON.parsefile(manifest_path)["source_archive_path"] ==
          summary["stages"][1]["archive_path"]

    # A committed rerun must be a read-only resume.  Hash every durable
    # artifact, not just the top-level summary, so duplicate stage writes or
    # advancement are observable.
    durable = String[]
    for (dir, _, files) in walkdir(summary["output_root"])
        append!(durable, joinpath.(dir, files))
    end
    before = Dict(path => bytes2hex(SHA.sha256(read(path))) for path in durable)
    rerun_proc = run_fixture_subprocess(
        `$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl $batch_path`,
        "resume/idempotence", batch_path; timeout_seconds=120.0)
    require_fixture_subprocess(rerun_proc, "resume/idempotence")
    rerun_summary = JSON.parse(rerun_proc["stdout"])
    @test rerun_summary == summary
    after = Dict(path => bytes2hex(SHA.sha256(read(path))) for path in durable)
    @test after == before

    # Removing a committed state artifact must fail closed rather than
    # treating the existing directory as completed work.
    rm(joinpath(summary["stages"][2]["stage_root"], "stage_state.json"))
    proc = run_fixture_subprocess(
        `$JULIA --project=$REPO $REPO/scripts/run_pipeline.jl $batch_path`,
        "expected-rejection/missing-state", batch_path; timeout_seconds=120.0)
    @test proc["status"] == "failed"
    @test proc["exit_code"] != 0
end
