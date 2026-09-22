#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_gt(path: Path, days: int) -> np.ndarray:
    values = np.zeros(days)
    with path.open() as handle:
        next(handle, None)
        for row in csv.reader(handle):
            try:
                day, value = int(float(row[0])), float(row[1])
            except (ValueError, IndexError):
                continue
            if 1 <= day <= days:
                values[day - 1] = value
    return values


def rolling_mean(values: np.ndarray, window: int = 7) -> np.ndarray:
    radius = window // 2
    return np.array([
        values[max(0, i - radius):min(len(values), i + radius + 1)].mean()
        for i in range(len(values))
    ])


def find_best_candidate(stage_dir: Path):
    best = None
    for metrics_path in stage_dir.glob("iter_*/cand_*/metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        score = metrics.get("score")
        if isinstance(score, (int, float)) and np.isfinite(score):
            if best is None or score < best[0]:
                best = (float(score), metrics_path.parent)
    if best is None:
        raise RuntimeError(f"No completed candidate found under {stage_dir}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    score, candidate_dir = find_best_candidate(args.stage_dir)
    config = json.loads((candidate_dir / "config.json").read_text())
    modulation = config["infection_modulation"]["params"]
    detection = config["mild_detection_modulation"]["params"]
    tracing = config["tracing_modulation"]["params"]
    with h5py.File(candidate_dir / "output_daily.jld2", "r") as handle:
        trajectories = [
            np.asarray(handle[key]["daily_detections"], dtype=float).ravel()
            for key in handle.keys()
        ]
    days = min(int(config["stop_simulation_time"]), min(len(values) for values in trajectories))
    gt = load_gt(args.gt_dir / "daily_age_total_detections.csv", days)
    trajectories = [values[:days] for values in trajectories]
    simulation = np.mean(np.asarray(trajectories), axis=0)
    timeline = np.arange(1, days + 1)

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [1, 1, 1, 2]}
    )
    for axis, params, title, color in [
        (axes[0], modulation, "Best infection_modulation", "#1976d2"),
        (axes[1], detection, "Best mild_detection_modulation", "#43a047"),
        (axes[2], tracing, "Best tracing_modulation", "#8e44ad"),
    ]:
        boundaries = params["interval_times"]
        values = params["interval_values"]
        if len(values) != len(boundaries) + 1:
            raise ValueError("MocosSim interval_values must have one more entry than interval_times")
        active = 1 + sum(boundary < days for boundary in boundaries)
        starts = [0] + boundaries[:active - 1]
        # Output day one starts at simulation t=0; each parameter has its own calendar.
        plot_days = [start + 1 for start in starts] + [days + 1]
        plot_values = values[:active] + [values[active - 1]]
        axis.step(plot_days, plot_values, where="post", lw=2, color=color)
        axis.fill_between(plot_days, plot_values, step="post", alpha=0.15, color=color)
        axis.set_xlim(1, days + 1)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Raw value")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)

    axes[3].bar(timeline, gt, color="#90caf9", alpha=0.45, width=0.9, label="GT raw")
    axes[3].plot(timeline, rolling_mean(gt), color="#1976d2", lw=2, label="GT 7d avg")
    axes[3].bar(
        timeline, simulation, color="#b0bec5", alpha=0.35, width=0.9,
        label="Simulation raw",
    )
    axes[3].plot(
        timeline, rolling_mean(simulation), color="#37474f", lw=2,
        ls="--", label="Simulation 7d avg",
    )
    axes[3].set_title(f"Detections: GT vs best simulation · score={score:.6f}")
    axes[3].set_xlabel("Simulation day")
    axes[3].set_ylabel("Detections")
    axes[3].grid(axis="y", alpha=0.3)
    axes[3].legend(loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved {args.out} from {candidate_dir}")


if __name__ == "__main__":
    main()
