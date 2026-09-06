# Portable Saxony production baseline

`optimizer_config.saxony.example.json` is the first production-oriented profile.
It uses repository-relative ground truth and output paths, while deliberately
requiring the following external inputs through environment variables:

| Variable | Required content |
|---|---|
| `JULIA_BIN` | Julia 1.7 executable used by MocosSimLauncher |
| `MOCOSSIM_LAUNCHER_DIR` | checkout of the launcher version compatible with this optimizer |
| `MOCOSSIM_ADVANCED_CLI` | `advanced_cli.jl` from that checkout |
| `MOCOSSIM_SEED_CONFIG` | complete Saxony seed JSON, including population, COVIMOD, and immunity JLD2 paths |

The source datasets and simulator are not redistributed by this repository.
Record their approved versions and licenses alongside each experiment manifest.
The example's calendar anchors day 1 provisionally at **2020-09-03** based on
the supplied seed naming. Confirm this date against upstream source metadata
before a scientific run; changing it changes every canonical observation date.

Run a validation-only preflight, which never launches the simulator or creates
the output root:

```sh
julia --project=. run_optimizer.jl --preflight optimizer_config.saxony.example.json
```

A production smoke run should first copy the profile, reduce the first stage to
`max_iterations: 1` and `population_size: 2`, and retain the resulting manifest.
The committed profile defines 3/6/12/18/24/30-month CMA-ES stages; the optional
NUTS step still operates after each stage on its archive-derived surrogate. Data
windows are independent of that optimizer choice. Every stage before the final
test horizon uses rolling-origin evaluation: its last 28 days are validation and
all earlier days in that stage are training. Thus the 3-month stage trains on
days 1–62 and validates on 63–90, the 6-month stage trains on 1–152 and validates
on 153–180, and the same rule continues through the 24-month stage.

The 30-month stage reaches the predeclared final evaluation horizon. It simulates
all 900 days, fits on days 1–760, selects on validation days 761–816, and leaves
817–900 as a frozen forecast test. “Reserved” therefore means “not visible to
CMA-ES/NUTS selection”, not “unused by the stage”: the simulator produces those
days so the selected model can be tested out of sample. The frozen test must not
be used to tune weights or select a candidate.

The Negative-Binomial weekly observation model is enabled in the example. It
uses metric-specific dispersion constants and is an initial auditable baseline,
not a claim that delay convolution, weekday effects, or reporting revisions are
already identified.

## Two-candidate production smoke test

The committed `optimizer_config.saxony.smoke.json` fixes the executable contract
to one 3-month iteration with exactly two candidates. Each adapter process has a
900-second deadline; a timeout terminates the process and leaves stdout, stderr,
exit status, and the exact command in `adapter_invocation.json`.

Set a fresh output directory and run locally:

```sh
export MOCOSSIM_SMOKE_OUTPUT="$PWD/runs/production-smoke-local"
julia --project=. scripts/run_production_smoke.jl optimizer_config.saxony.smoke.json
```

The command fails unless both candidates finish and every `output_daily.jld2`
contains finite, non-empty detections, deaths, and hospitalization trajectories.
On success it writes `production_smoke_manifest.json` with optimizer and launcher
commits, input hashes, environment details, RNG seeds, commands, candidate
terminal states, and JLD2 schema reports. See
[`examples/production-smoke-manifest.json`](examples/production-smoke-manifest.json)
for the expected shape.

Run the equivalent Slurm contract into a different directory:

```sh
export MOCOSSIM_SMOKE_OUTPUT="$PWD/runs/production-smoke-slurm"
julia --project=. scripts/run_production_smoke.jl --slurm optimizer_config.saxony.smoke.json
```

Then compare the execution contracts. Numerical scores are deliberately not
required to be identical because the simulator may be stochastic; revisions,
inputs, stage shape, required metrics, and trajectory counts must match.

```sh
julia --project=. scripts/run_production_smoke.jl --compare \
  runs/production-smoke-local/production_smoke_manifest.json \
  runs/production-smoke-slurm/production_smoke_manifest.json
```

The integration test runs the real local adapter whenever all four external
input variables are present. It is skipped only when those external inputs are
not supplied; JLD2 validation and local/Slurm parity contract tests always run.
