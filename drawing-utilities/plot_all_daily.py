#!/usr/bin/env python3
"""Generate model_vs_gt.png next to every daily.jld2 found under search_local."""

import argparse, os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

GT_DIR = "/Users/marcinbodych/Workspace/saxocov/LLM-IDEAS/manager/runs/gt"
TOTAL_DAYS = 112

def read_gt_csv(path):
    values = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                values.append(float(row[1]))
            except (IndexError, ValueError):
                pass
    return np.array(values)

def read_metric(path, metric):
    arrays = []
    try:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                grp = f[key]
                if metric in grp:
                    arrays.append(np.array(grp[metric], dtype=float))
    except Exception:
        return None
    if not arrays:
        return None
    max_len = max(len(a) for a in arrays)
    padded = np.zeros((len(arrays), max_len))
    for i, a in enumerate(arrays):
        padded[i, :len(a)] = a
    return padded.mean(axis=0)

def rolling_avg(arr, window=7):
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = arr[start:i+1].mean()
    return result

def pad(arr, length):
    if arr is None:
        return np.zeros(length)
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.zeros(length - len(arr))])

GT = {
    "daily_detections":       read_gt_csv(os.path.join(GT_DIR, "daily_detections.csv")),
    "daily_deaths":           read_gt_csv(os.path.join(GT_DIR, "daily_deaths.csv")),
    "daily_hospitalizations": read_gt_csv(os.path.join(GT_DIR, "daily_hospitalizations.csv")),
}
METRICS = list(GT.keys())
TITLES  = ["Daily Detections", "Daily Deaths", "Daily Hospitalizations"]
COLORS  = ["#2196F3", "#F44336", "#FF9800"]
MB      = [1, 29, 57, 85, 113]

def make_plot(daily_path, out_path):
    days = np.arange(1, TOTAL_DAYS + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # title from path: month_XX/week_XX/iter_XX_cXX
    parts = daily_path.replace(os.sep, "/").split("/")
    try:
        label = "/".join(parts[-4:-1])
    except Exception:
        label = daily_path
    fig.suptitle(label, fontsize=10)

    mc = ["#E3F2FD", "#FCE4EC", "#F3E5F5", "#E8F5E9"]

    for ax, metric, title, color in zip(axes, METRICS, TITLES, COLORS):
        pred_raw = read_metric(daily_path, metric)
        pred = pad(pred_raw, TOTAL_DAYS)
        gt   = pad(GT[metric], TOTAL_DAYS)

        for i, (lo, hi) in enumerate(zip(MB[:-1], MB[1:])):
            ax.axvspan(lo, hi, alpha=0.12, color=mc[i], zorder=0)
        for b in MB[1:-1]:
            ax.axvline(b, color="#aaa", lw=0.7, ls=":")

        ax.bar(days, gt,   color=color,   alpha=0.2, width=1.0)
        ax.plot(days, rolling_avg(gt),   color=color,   lw=1.8, label="GT")
        ax.bar(days, pred, color="#607D8B", alpha=0.15, width=1.0)
        ax.plot(days, rolling_avg(pred), color="#263238", lw=1.8, ls="--", label="Model")

        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(1, TOTAL_DAYS)

    axes[-1].set_xlabel("Simulation day", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", default=
        "/Users/marcinbodych/Workspace/saxocov/experiment-wcss/LLM-IDEAS/manager/runs/search_local")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.search_dir, "**", "daily.jld2"), recursive=True))
    print(f"Found {len(files)} daily.jld2 files")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks = []
    for f in files:
        out = os.path.join(os.path.dirname(f), "model_vs_gt.png")
        tasks.append((f, out))

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(make_plot, f, o): (f, o) for f, o in tasks}
        for fut in as_completed(futs):
            f, o = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  FAILED {f}: {e}")
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} done")

    print(f"Done — {len(tasks)} plots written.")

if __name__ == "__main__":
    main()
