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

## Twelve-month production pilot

`optimizer_config.saxony.12m-pilot.json` is the recommended first production run.
It deliberately stops at 12 months, so a promising rolling-validation result can
be reviewed before any 18/24/30-month simulations are funded. A 9-month stage
provides an intermediate transfer instead of asking the 12-month stage to absorb
the entire six-month horizon increase. The first three stages use four iterations
each, giving CMA-ES more than two or three covariance updates before a stage
transition; the target 12-month stage uses five. The resulting upper bound is
344 CMA candidates. The longer stages use 24 candidates per generation, which
keeps some population diversity without paying for 50 simulations at every step:

```text
(4 × 16) + (4 × 16) + (4 × 24) + (5 × 24) = 344
```

This is 5.39% of the 6,384-candidate full profile. Three configured final
validation seeds bring the expected simulator invocation count to 347, assuming
no retries. Use a fresh `runs/saxony-12m-pilot` output directory and retain
the preflight and stage manifests. Five generations at 12 months are intended to
provide a first go/no-go signal, not by themselves to establish convergence.

### Pilot parameter audit

| Setting | Pilot value | Reason |
|---|---:|---|
| Initial `sigma` | `0.12` | This is the implementation's maximum sigma component. With an identity covariance it is a 12%-of-range marginal standard deviation; after covariance learning or transfer, the actual coordinate deviation is `sigma[i] * sqrt(covariance[i,i])`. Larger configured values are clamped and are therefore misleading. |
| Scalar preprocessing | normalized to `[0, 1]` | Puts the three scalar ranges on the same coordinate scale as the temporal modulation bounds, so one `sigma` has a consistent interpretation. |
| Population | `16/16/24/24` | Exceeds the usual small CMA population heuristic for the active 12/21/30/39-dimensional stages while remaining affordable. |
| Iterations | `4/4/4/5` | Enough for a pilot trend and staged transfer check, but explicitly not a convergence claim. |
| Completion | `1.0`, zero grace delay | Avoids updating CMA from a scheduler-dependent subset and avoids an unnecessary delay after all candidates finish. |
| Adapter/iteration timeout | `3300/14400` seconds | The adapter remains bounded below the 75-minute task walltime; the controller allows queue latency rather than cancelling an array merely because it waited more than an hour to start. |
| Archive continuation | at least 5 finite entries, score below `1e6` | A permissive operational guard that allows a small pilot archive to transfer; it is not a scientific quality threshold. Review every `stage_extension_gate.json`. |
| NUTS surrogate | disabled | Avoids fitting and transferring a surrogate posterior from only four or five CMA generations; it also reduces controller overhead. |

The population heuristic is only a sanity check, not proof that covariance is
well estimated. At 12 months the search has 39 active coordinates (three scalar
and three sets of twelve monthly values); five generations cannot identify a
full 39-dimensional covariance reliably. The pilot should therefore be judged
on score/validation trends, failure rate, bound hits, and stability across the
three final seeds. Increase iterations only after those artifacts justify the
extra simulator budget.

### Sigma audit by horizon

There is no defensible horizon-only formula for an “optimal” sigma. Sigma is an
initial scale, while CMA adapts a vector of coordinate-wise scales and a full
covariance from ranked candidates. Its appropriate value depends on the basin
around the transferred mean, simulator noise, clipping, and the remaining
generation budget—not on whether the observations cover 3, 6, 9, or 12 months.

For the **currently budgeted** `4/4/4/5`-generation pilot, retain `0.12` at all
four stages. This is a deliberate exploration-biased choice, not an estimate of
the optimum:

| Horizon | Active dimensions | Generations | Recommended configured sigma | Interpretation |
|---|---:|---:|---:|---|
| 3 months | 12 | 4 | `0.12` | Starts from the seed with the widest permitted normalized search. |
| 6 months | 21 | 4 | `0.12` | Gives newly introduced temporal coordinates enough movement in a four-update stage. |
| 9 months | 30 | 4 | `0.12` | Preserves exploration because four updates cannot reliably tune a 30-dimensional distribution. |
| 12 months | 39 | 5 | `0.12` | Appropriate for a go/no-go exploration pass, but not evidence of convergence or optimal tuning. |

These values must not be read as four independently applied marginal standard
deviations. On a stage transition, inherited coordinates carry their learned
sigma/covariance and are expanded by transition settings; new coordinates get
transition uncertainty. The configured stage sigma participates in floors and
initialization, and every sigma component is finally clamped to `[0.02, 0.12]`.
Consequently, changing only the later-stage JSON values does not define the
actual sampling radius.

For the recommended corrected run of at least 20 generations, do not declare a
single sigma in advance. Run a matched-seed, equal-candidate-budget tournament
from the same effective predecessor state with initial scales `0.06`, `0.09`,
and `0.12`. Select using rolling-origin validation loss (with a noise penalty),
and report the actual marginal deviation, clipping/bound-hit rate, and diversity
alongside sigma. A useful decision rule is to reject a scale that clips heavily
in its first two generations, or one whose population spread collapses before
validation loss improves. This experiment, rather than the corrupted historical
run or horizon length alone, is what can identify a better initial scale.

### Why the individual metric weights are zero

The pilot uses one coherent **joint weekly Negative-Binomial loss** for candidate
ranking. Its configured likelihood dimensions are detections and deaths; each
weekly observation from those two streams contributes once, using explicit
dispersion values of 25 and 10 respectively. Hospitalizations remain required as
an output-integrity and diagnostic stream, but do not rank candidates because
their nominal comparability is weaker. The stored
`negative_binomial_log_likelihood` metric is the negative of that joint log
likelihood, so minimizing it is equivalent to maximizing the probability of the
observed counts under the model.

In the mean/dispersion parameterization used here, a count with predicted mean
`mu` and dispersion `r` has variance `mu + mu^2/r`. A larger `r` therefore means
less assumed extra-Poisson noise and makes a discrepancy more informative; it is
not a direct percentage importance weight. Thus `r=25` says detections are
assumed less overdispersed than deaths at `r=10`. Deaths still contribute every
week, but the model tolerates more relative count variability in that stream.
These values are auditable pilot assumptions, not fitted truths.

The individual daily, cumulative, blocked-cumulative, and `weekly_control`
weights are zero because those values are retained as diagnostics. Giving them
positive weights as well would count the same observations multiple times in
incompatible units: once through the count likelihood, again through relative
errors, and again through cumulative errors. Hospitalizations are excluded due
to weaker nominal comparability, while `daily_student_detections` is excluded
because it is sparse. Both remain available for reporting.

The two nonzero values outside the likelihood are regularizers, not additional
observation fits: `temporal_jump_weight = 0.5` discourages jagged adjacent monthly
modulations, and `infection_extrema_weight = 0.1` weakly discourages excessive
direction changes and boundary hits. Thus the effective objective is:

```text
negative weekly NB log likelihood
  + 0.5 × temporal jump penalty
  + 0.1 × infection extrema penalty
```

The explicit `validation.likelihood_metrics` and `likelihood_dispersions`
objects prevent hospitalization, optional age, or student series present in
`gt/` from silently entering the joint likelihood and make its noise assumptions
visible in the experiment configuration.

After the local and Slurm smoke checks below pass, submit the budgeted profile:

```sh
export JULIA_BIN=/path/to/julia
export MOCOSSIM_LAUNCHER_DIR=/path/to/MocosSimLauncher
export MOCOSSIM_ADVANCED_CLI="$MOCOSSIM_LAUNCHER_DIR/advanced_cli.jl"
export MOCOSSIM_SEED_CONFIG=/path/to/saxony-seed.json
sbatch scripts/run_cmaes.slurm optimizer_config.saxony.12m-pilot.json
```

The wrapper performs a no-launch preflight before starting the controller. It
also defaults to the 12-month pilot when no positional config is supplied.
The controller reserves one CPU while it coordinates four-CPU array tasks. Julia
environments are instantiated once by the wrapper, not concurrently by every
candidate. Candidate plotting is disabled by default; set
`MOCOSSIM_PLOT_CANDIDATES=1` only when the Python plotting dependencies have
already been installed on compute nodes.

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

The integration contract always runs a hermetic two-candidate smoke through the
installed Julia executable and the same local adapter subprocess boundary. This
keeps command construction, candidate terminal states, manifest persistence, and
90-day JLD2 validation covered in automation without redistributing production
data. The additional Saxony run uses the real launcher whenever all four external
input variables are present; only that data-dependent check is skipped when the
external inputs are not supplied. Local/Slurm parity checks always run as well.

## Corrected 6m–9m–12m tournament

`optimizer_config.saxony.corrected.json` is a fresh-state rerun profile. Point
`MOCOSSIM_SEED_CONFIG` at the accepted three-month effective candidate; never at
a `stage_state.json` or reusable state from the affected run. Each stage searches
only the three normalized scalars and the three newly introduced months for each
of the three temporal series (12 dimensions total), while `vector_to_config`
preserves the historical temporal prefix.

Prepare three independent, equal-budget policies with:

```bash
julia --project=. scripts/run_corrected_tournament.jl \
  optimizer_config.saxony.corrected.json --prepare-only
```

Remove `--prepare-only` to run locally, or add `--slurm`. The launcher refuses to
reuse an existing policy output directory. Baseline and temporal-escape share
optimizer seed 42; restart uses optimizer seed 1042, while all policies retain
the identical simulator validation seeds. The stages have 10, 15, and 20 maximum
generations and stop only after at least six generations when both best and
median validation loss improve by less than 1% for three generations.

Candidate ranking uses the rolling-origin validation loss; the training
likelihood remains stored as `training_score`. Every generation re-evaluates the
selection half of the population with seeds 43 and 44 and ranks it by mean
validation loss plus one standard error; the initial evaluation uses the same
seed for every candidate. Extension requires at least eight
finite archive entries under the stage-relative validation threshold of 1.0 and
a diversity pass. Final status is `validation_failed`, with `run_failed.json`,
unless all three configured validation replicates are finite. Adapter rejection
metadata retains `output_error` so an exit-zero invalid output is diagnosable.
