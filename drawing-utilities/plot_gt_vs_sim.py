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
import matplotlib.dates as mdates
import numpy as np
import h5py


def load_sparse_series(gt_dir: Path, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    path = gt_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing GT file: {path}")
    days: List[int] = []
    values: List[float] = []
    with path.open() as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                days.append(int(float(parts[0])))
                values.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(days, dtype=int), np.array(values, dtype=float)


def load_weekly_series(gt_dir: Path, filename: str, total_days: int) -> np.ndarray:
    days, values = load_sparse_series(gt_dir, filename)
    out = np.zeros(total_days, dtype=float)
    for day, value in zip(days, values):
        if 1 <= day <= total_days:
            out[day - 1] = value
    return out


def load_gt(gt_dir: Path) -> Dict[str, np.ndarray]:
    gt = {}
    files = {
        "detections": gt_dir / "daily_detections.csv",
        "deaths": gt_dir / "daily_deaths.csv",
        "hospitalizations": gt_dir / "daily_hospitalizations.csv",
        "student_detections": gt_dir / "daily_student_detections.csv",
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
    student_detections = []
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
        student_detections.append(signal * 0.35)
        hospitalizations.append(signal * 0.07)  # simple scaling for visualization
        deaths.append(signal * 0.01)            # simple scaling for visualization

    return {
        "detections": np.array(detections, dtype=float),
        "student_detections": np.array(student_detections, dtype=float),
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


def rolling_mean(arr: np.ndarray, window: int = 7) -> np.ndarray:
    if len(arr) == 0:
        return arr
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo:i+1].mean()
    return out


def rolling_max(arr: np.ndarray, window: int = 7) -> np.ndarray:
    if len(arr) == 0:
        return arr
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = np.max(arr[lo:i+1])
    return out


def sparse_cumulative(days: np.ndarray, values: np.ndarray, total_days: int) -> np.ndarray:
    out = np.zeros(total_days, dtype=float)
    acc = 0.0
    idx = 0
    for day in range(1, total_days + 1):
        while idx < len(days) and days[idx] == day:
            acc += values[idx]
            idx += 1
        out[day - 1] = acc
    return out


def day_to_date(start: np.datetime64, day: int) -> np.datetime64:
    return start + np.timedelta64(day - 1, "D")


def date_range(start: np.datetime64, total_days: int) -> np.ndarray:
    return np.array([day_to_date(start, d) for d in range(1, total_days + 1)], dtype="datetime64[D]")


def cumulative(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr, dtype=float)


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


def plot(gt: Dict[str, np.ndarray], sim: Dict[str, np.ndarray], out_path: Path, gt_dir: Path, stop_day: int):
    metrics = [("detections", "Daily Detections", "#2196F3"),
               ("student_detections", "Daily Student Detections", "#4CAF50"),
               ("hospitalizations", "7-day Hospitalizations (roll sum)", "#FF9800"),
               ("deaths", "Daily Deaths", "#F44336")]
    gt_avg_labels = {
        "student_detections": "GT 7d max",
    }

    total_days = min(max(len(gt[m]) for m, *_ in metrics), stop_day)
    start_date = np.datetime64("2020-09-01")
    dates = date_range(start_date, total_days)
    date_numbers = mdates.date2num([np.datetime64(d, "D").astype(object) for d in dates])

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(hspace=0.25)
    fig.suptitle("Ground Truth vs Synthetic Simulation", fontsize=12)

    for ax, (key, title, color) in zip(axes, metrics):
        g = load_weekly_series(gt_dir, "daily_student_detections.csv", total_days) if key == "student_detections" else pad(gt[key], total_days)
        s = pad(sim[key], total_days)
        if key == "hospitalizations":
            s = rolling_mean(s, 7)
        ax.bar(date_numbers, g, color=color, alpha=0.25, width=0.9, label="GT raw")
        if key == "student_detections":
            ax.plot(date_numbers, rolling_max(g), color=color, lw=2, label="GT 7d max")
        else:
            gt_label = gt_avg_labels.get(key, "GT 7d avg")
            ax.plot(date_numbers, rolling_avg(g), color=color, lw=2, label=gt_label)
        ax.bar(date_numbers, s, color="#607D8B", alpha=0.18, width=0.9, label="Sim raw")
        ax.plot(date_numbers, rolling_avg(s), color="#37474F", lw=2, ls="--", label="Sim 7d avg")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig2.subplots_adjust(hspace=0.25)
    fig2.suptitle("Ground Truth vs Synthetic Simulation (Cumulative)", fontsize=12)
    for ax, (key, title, color) in zip(axes2, metrics):
        if key == "student_detections":
            sparse_days, sparse_vals = load_sparse_series(gt_dir, "daily_student_detections.csv")
            g = sparse_cumulative(sparse_days, sparse_vals, total_days)
        else:
            g = cumulative(pad(gt[key], total_days))
        s = cumulative(pad(sim[key], total_days))
        ax.plot(date_numbers, g, color=color, lw=2, label="GT cumulative")
        ax.plot(date_numbers, s, color="#37474F", lw=2, ls="--", label="Sim cumulative")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    for ax in list(axes) + list(axes2):
        ax.set_xlim(date_numbers[0], date_numbers[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", alpha=0.15)
    axes[-1].set_xlabel("Date", fontsize=10)
    axes2[-1].set_xlabel("Date", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    cum_path = out_path.with_name(out_path.stem + "_cumulative" + out_path.suffix)
    fig2.tight_layout()
    fig2.savefig(cum_path, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path}")
    print(f"Saved: {cum_path}")


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
        plain_cfg = base / "config.json"
        if cand_cfg.exists():
            cfg_path = cand_cfg
        elif final_cfg.exists():
            cfg_path = final_cfg
        elif plain_cfg.exists():
            cfg_path = plain_cfg

    cfg = None
    if cfg_path is not None and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    elif args.daily is None:
        raise SystemExit("Config JSON not found (config_candidate.json/final_best_candidate.json/config.json) and no --daily provided")
    gt = load_gt(args.gt_dir.resolve())
    max_days = args.days or len(gt["detections"])

    if args.daily:
        sim_detections = read_daily_metric(str(args.daily), "daily_detections")
        sim_hosp = read_daily_metric(str(args.daily), "daily_hospitalizations")
        sim_deaths = read_daily_metric(str(args.daily), "daily_deaths")
        sim_student_detections = read_daily_metric(str(args.daily), "daily_student_detections")
        if sim_detections is None:
            raise SystemExit(f"Cannot read daily_detections from {args.daily}")
        sim = {
            "detections": np.array(sim_detections[:max_days], dtype=float),
            "hospitalizations": np.array(sim_hosp[:max_days], dtype=float) if sim_hosp is not None else np.zeros(max_days, dtype=float),
            "deaths": np.array(sim_deaths[:max_days], dtype=float) if sim_deaths is not None else np.zeros(max_days, dtype=float),
            "student_detections": np.array(sim_student_detections[:max_days], dtype=float) if sim_deaths is not None else np.zeros(max_days, dtype=float),
        }
    else:
        sim = synthetic_simulation(cfg, max_days)

    out_path = args.out
    if out_path is None:
        if args.daily:
            out_path = args.daily.with_suffix(".gt_vs_sim.png")
        else:
            out_path = (base / "plots" / "gt_vs_sim.png")
    stop_day = int(cfg.get("stop_simulation_time", max_days)) if cfg is not None else max_days
    plot(gt, sim, Path(out_path), args.gt_dir.resolve(), stop_day)


if __name__ == "__main__":
    main()
