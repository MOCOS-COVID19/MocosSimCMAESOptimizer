#!/usr/bin/env python3
"""Generate an interactive pipeline index.html from saved CMA/NUTS artifacts."""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * q
    i = int(x)
    j = min(i + 1, len(values) - 1)
    return values[i] + (values[j] - values[i]) * (x - i)


def stage_history(stage_dir):
    rows = []
    path = stage_dir / "iter_metrics.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row.get("score"), (int, float)) and math.isfinite(row["score"]):
                rows.append(row)
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["iteration"]), []).append(float(row["score"]))
    return [
        {"iteration": iteration, "best": min(values), "mean": statistics.mean(values), "n": len(values), "completed": len(values)}
        for iteration, values in sorted(grouped.items())
    ]


def sigma_history(stage_dir):
    result = []
    for path in stage_dir.glob("iter_*/cma_sampling_state.json"):
        if not path.parent.name[5:].isdigit():
            continue
        state = load(path)
        values = state.get("sigma", [])
        values = values if isinstance(values, list) else [values]
        values = [float(value) for value in values]
        result.append({"iteration": int(path.parent.name[5:]), "min": min(values), "mean": statistics.mean(values), "max": max(values), "n": len(values)})
    return sorted(result, key=lambda row: row["iteration"])


def candidate_data(stage_dir, relative_prefix):
    candidates = []
    for metrics_path in stage_dir.glob("iter_*/cand_*/metrics.json"):
        metrics = load(metrics_path)
        score = metrics.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        iteration = int(metrics_path.parent.parent.name[5:])
        candidate = int(metrics_path.parent.name[5:])
        candidates.append({
            "iteration": iteration,
            "candidate": candidate,
            "score": float(score),
            "simulated": metrics.get("simulated"),
            "image": f"{relative_prefix}/iter_{iteration}/cand_{candidate:02d}/gt_vs_sim.png",
            "cumulative": f"{relative_prefix}/iter_{iteration}/cand_{candidate:02d}/gt_vs_sim_cumulative.png",
        })
    return sorted(candidates, key=lambda row: row["score"])


def modulation_data(stage_dir):
    result = []
    for path in stage_dir.glob("iter_*/cma_sampling_state.json"):
        if not path.parent.name[5:].isdigit():
            continue
        state = load(path)
        values = [float(value) for name, value in zip(state["param_names"], state["mean"]) if name.startswith("infection_modulation.params.interval_values[")]
        if values:
            result.append({"iteration": int(path.parent.name[5:]), "values": values})
    return sorted(result, key=lambda row: row["iteration"])


def posterior_data(path):
    if not path.exists():
        return [], {}
    posterior = load(path)
    samples = posterior.get("samples", [])
    names = posterior.get("param_names", [])
    scales = posterior.get("scale", [])
    result = []
    for index, name in enumerate(names):
        values = [float(row[index]) for row in samples if index < len(row) and isinstance(row[index], (int, float))]
        result.append({"name": name, "mean": statistics.mean(values) if values else None, "q05": quantile(values, .05), "q50": quantile(values, .5), "q95": quantile(values, .95), "scale": scales[index] if index < len(scales) else None})
    info = {key: posterior.get(key) for key in ["sampler", "draws", "warmup", "max_depth", "step_size", "temperature", "seed"]}
    return result, info


def make_scalar_plot(stage_dir, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; scalar plot not generated", file=sys.stderr)
        return
    params = ["class", "school", "age_coupling_param"]
    labels = {"class": "class kernel", "school": "school kernel", "age_coupling_param": "age coupling kernel"}
    colors = {"class": "#e67e22", "school": "#8e44ad", "age_coupling_param": "#1976d2"}
    series = {param: [] for param in params}
    for iteration_dir in sorted([path for path in stage_dir.glob("iter_*") if path.is_dir() and path.name[5:].isdigit()], key=lambda path: int(path.name[5:])):
        rows = []
        for config_path in iteration_dir.glob("cand_*/config.json"):
            metrics_path = config_path.parent / "metrics.json"
            if not metrics_path.exists():
                continue
            score = load(metrics_path).get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                continue
            transmission = load(config_path)["transmission_probabilities"]
            rows.append((float(score), {param: float(transmission[param]) for param in params}))
        if rows:
            rows.sort()
            for param in params:
                series[param].append((int(iteration_dir.name[5:]), rows[0][1][param], statistics.mean(row[1][param] for row in rows)))
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, param in zip(axes, params):
        values = series[param]
        axis.plot([row[0] for row in values], [row[1] for row in values], marker="o", lw=2, color=colors[param], label="best candidate")
        axis.plot([row[0] for row in values], [row[2] for row in values], marker=".", lw=1.5, ls="--", color="#455a64", label="population mean")
        axis.set_title(labels[param])
        axis.set_ylabel("value")
        axis.grid(axis="y", alpha=.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("short-stage iteration")
    figure.suptitle("Scalar transmission parameter evolution")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("index_template.html"))
    parser.add_argument("--gt-dir", type=Path, default=Path(__file__).resolve().parents[1] / "gt")
    args = parser.parse_args()
    root = args.pipeline_root.resolve()
    stage_dirs = sorted(
        [path for path in root.glob("*/real_sims/*") if path.is_dir()],
        key=lambda path: path.as_posix(),
    )
    if not stage_dirs:
        raise FileNotFoundError(f"No stage simulation directories found under {root}")
    short_dir = stage_dirs[0]
    long_dir = stage_dirs[-1]
    short_stage = short_dir.parents[1].name
    long_stage = long_dir.parents[1].name
    short_summary = load(short_dir / "stage_state.json")
    long_summary = load(long_dir.parent.parent / f"{long_stage}_summary.json") if (long_dir.parent.parent / f"{long_stage}_summary.json").exists() else {}
    short_posterior_path = short_dir / "posterior_samples.json"
    long_posterior_path = long_dir / "posterior_samples.json"
    posterior_path = long_posterior_path if long_posterior_path.exists() else short_posterior_path
    posterior, posterior_info = posterior_data(posterior_path)
    short_candidates = candidate_data(short_dir, f"{short_stage}/real_sims/{short_dir.name}")
    long_candidates = candidate_data(long_dir, f"{long_stage}/real_sims/{long_dir.name}")
    for candidate in short_candidates:
        candidate["stage"] = short_stage
    for candidate in long_candidates:
        candidate["stage"] = long_stage
    candidates = sorted(short_candidates + long_candidates, key=lambda row: row["score"])
    data = {
        "pipeline": {},
        "short": {"best": short_summary.get("best_score"), "stage": short_stage, "sigma": short_summary.get("sigma")},
        "long": {"best": long_summary.get("best_score"), "stage": long_stage, "sigma": long_summary.get("sigma")},
        "history": stage_history(short_dir),
        "sigmas": sigma_history(short_dir),
        "posterior": posterior,
        "posteriorInfo": posterior_info,
        "iterations": stage_history(short_dir),
        "candidates": candidates,
        "total_candidates": len(candidates),
        "dimensions": len(posterior),
        "paramCount": len(posterior),
        "infection": modulation_data(short_dir),
    }
    make_scalar_plot(short_dir, root / short_stage / "scalar_parameters_evolution.png")
    modulation_plotter = Path(__file__).with_name("plot_best_modulation_detections.py")
    if modulation_plotter.exists() and args.gt_dir.exists():
        for stage_dir in sorted(set([short_dir, long_dir])):
            output = stage_dir.parent.parent / "infection_modulation_best.png"
            subprocess.run(
                [
                    sys.executable,
                    str(modulation_plotter),
                    "--stage-dir", str(stage_dir),
                    "--gt-dir", str(args.gt_dir),
                    "--out", str(output),
                ],
                check=False,
            )
    template = args.template.read_text()
    output = (
        template
        .replace("short_6m", short_stage)
        .replace("long_12m", long_stage)
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__PIPELINE_NAME__", root.name)
    )
    (root / "index.html").write_text(output)
    print(f"generated {root / 'index.html'}")


if __name__ == "__main__":
    main()
