#!/usr/bin/env python3
"""
Plot ground truth vs simulated daily series for detections, hospitalizations, and deaths.

Inputs:
- Optimizer outputs from MocosSimCMAESOptimizer under runs/default (or --output-dir):
    * final_best_candidate.json (configuration used)
    * stage_*_best_candidate.json (optional; not directly used)
- Ground truth CSVs with header, two columns: day,value
    * daily_detections.csv
    * daily_hospitalizations.csv
    * daily_deaths.csv

Because the Julia optimizer here runs a synthetic simulation inside Julia and does
not emit daily trajectories, this script visualizes the synthetic series produced
by the same scoring function to overlay with provided GT.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py


def load_gt(gt_dir: Path) -> Dict[str, np.ndarray]:
    gt = {}
    files = {
        "detections": gt_dir / "daily_detections.csv",
        "deaths": gt_dir / "daily_deaths.csv",
        "hospitalizations": gt_dir / "daily_hospitalizations.csv",
    }
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing GT file: {path}")
        values: List[float] = []
        with path.open() as f:
            next(f, None)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                try:
                    values.append(float(parts[1]))
                except ValueError:
                    continue
        gt[key] = np.array(values, dtype=float)
    return gt


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def synthetic_simulation(config: Dict, days: int) -> Dict[str, np.ndarray]:
    """Replicate the Julia scoring synthetic_simulation for plotting."""
    infection = list(map(float, config["infection_modulation"]["params"]["interval_values"]))
    detection = list(map(float, config["mild_detection_modulation"]["params"]["interval_values"]))
    tracing   = list(map(float, config["tracing_modulation"]["params"]["interval_values"]))
    household = float(config["transmission_probabilities"]["household"])
    school    = float(config["transmission_probabilities"]["school"])
    classv    = float(config["transmission_probabilities"]["class"])
    agec      = float(config["transmission_probabilities"]["age_coupling_param"])
    mild      = float(config["mild_detection_prob"])
    hosp      = float(config["initial_conditions"]["hospitalization_multiplier"])
    precision = float(config["screening"]["precision"])
    trace_prob = float(config["household_params"]["trace_prob"])
    quarantine_prob = float(config["household_params"]["quarantine_prob"])

    detections = []
    # For this synthetic visual, derive hospitalizations/deaths as scaled detections
    hospitalizations = []
    deaths = []

    for day in range(1, days + 1):
        bucket = min((day + 29) // 30, len(infection))  # 1-based buckets per 30 days
        base = 15.0 * infection[bucket - 1] * (0.6 + mild * detection[bucket - 1])
        contact = 50.0 * (0.8 * household + 0.4 * school + 0.7 * classv + 0.5 * agec)
        control = 10.0 * tracing[bucket - 1] * (trace_prob + quarantine_prob + precision)
        trend = 0.12 * day * infection[bucket - 1]
        signal = max(0.0, base + contact + trend - control + 6.0 * hosp)
        detections.append(signal)
        hospitalizations.append(signal * 0.07)  # simple scaling for visualization
        deaths.append(signal * 0.01)            # simple scaling for visualization

    return {
        "detections": np.array(detections, dtype=float),
        "hospitalizations": np.array(hospitalizations, dtype=float),
        "deaths": np.array(deaths, dtype=float),
    }


def pad(series: np.ndarray, length: int) -> np.ndarray:
    if series is None:
        return np.zeros(length)
    if len(series) >= length:
        return series[:length]
    return np.concatenate([series, np.zeros(length - len(series))])


def rolling_avg(arr: np.ndarray, window: int = 7) -> np.ndarray:
    if len(arr) == 0:
        return arr
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo:i+1].mean()
    return out


def read_daily_metric(path: str, metric: str):
    try:
        with h5py.File(path, "r") as f:
            vals = []
            for key in f.keys():
                grp = f[key]
                if metric in grp:
                    data = np.array(grp[metric], dtype=float)
                    vals.append(data)
            if not vals:
                return None
            max_len = max(len(v) for v in vals)
            padded = np.zeros((len(vals), max_len))
            for i, v in enumerate(vals):
                padded[i, : len(v)] = v
            return padded.mean(axis=0)
    except Exception:
        return None


def plot(gt: Dict[str, np.ndarray], sim: Dict[str, np.ndarray], out_path: Path):
    metrics = [("detections", "Daily Detections", "#2196F3"),
               ("hospitalizations", "7-day Hospitalizations (roll sum)", "#FF9800"),
               ("deaths", "Daily Deaths", "#F44336")]

    total_days = max(len(gt[m]) for m, *_ in metrics)
    days = np.arange(1, total_days + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Ground Truth vs Synthetic Simulation", fontsize=12)

    for ax, (key, title, color) in zip(axes, metrics):
        g = pad(gt[key], total_days)
        s = pad(sim[key], total_days)
        if key == "hospitalizations":
            s = np.convolve(s, np.ones(7), "full")[:total_days]
        ax.bar(days, g, color=color, alpha=0.25, width=1.0, label="GT raw")
        ax.plot(days, rolling_avg(g), color=color, lw=2, label="GT 7d avg")
        ax.bar(days, s, color="#607D8B", alpha=0.18, width=1.0, label="Sim raw")
        ax.plot(days, rolling_avg(s), color="#37474F", lw=2, ls="--", label="Sim 7d avg")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Day", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory with config_candidate.json or final_best_candidate.json; default: runs/default")
    parser.add_argument("--config", type=Path, default=None,
                        help="Explicit config JSON (overrides output-dir lookup)")
    parser.add_argument("--daily", type=Path, default=None,
                        help="Optional daily.jld2 to plot real sim series (if provided, use it instead of synthetic)")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Directory with daily_detections.csv, daily_hospitalizations.csv, daily_deaths.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Path to save the plot (default: <output-dir>/plots/gt_vs_sim.png or <daily>.png)")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit to first N days (default: length of GT detections)")
    args = parser.parse_args()

    base = args.output_dir or Path("runs/default")
    cfg_path = args.config
    if cfg_path is None:
        cand_cfg = base / "config_candidate.json"
        final_cfg = base / "final_best_candidate.json"
        if cand_cfg.exists():
            cfg_path = cand_cfg
        elif final_cfg.exists():
            cfg_path = final_cfg
    if cfg_path is None or not cfg_path.exists():
        raise SystemExit("Config JSON not found (config_candidate.json or final_best_candidate.json)")

    cfg = json.loads(cfg_path.read_text())
    gt = load_gt(args.gt_dir.resolve())
    max_days = args.days or len(gt["detections"])

    if args.daily:
        sim_detections = read_daily_metric(str(args.daily), "daily_detections")
        sim_hosp = read_daily_metric(str(args.daily), "daily_hospitalizations")
        sim_deaths = read_daily_metric(str(args.daily), "daily_deaths")
        if sim_detections is None:
            raise SystemExit(f"Cannot read daily_detections from {args.daily}")
        sim = {
            "detections": np.array(sim_detections[:max_days], dtype=float),
            "hospitalizations": np.array(sim_hosp[:max_days], dtype=float) if sim_hosp is not None else np.zeros(max_days, dtype=float),
            "deaths": np.array(sim_deaths[:max_days], dtype=float) if sim_deaths is not None else np.zeros(max_days, dtype=float),
        }
    else:
        sim = synthetic_simulation(cfg, max_days)

    out_path = args.out
    if out_path is None:
        if args.daily:
            out_path = args.daily.with_suffix(".gt_vs_sim.png")
        else:
            out_path = (base / "plots" / "gt_vs_sim.png")
    plot(gt, sim, Path(out_path))


if __name__ == "__main__":
    main()
