"""Plot RMSE history saved by the manager."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    monitor_path = root / "runs" / "rsme_monitor.json"
    plot_path = root / "plots" / "rsme_progress.png"
    if not monitor_path.exists():
        raise SystemExit("RMSE monitor file not found; run the manager first")
    data = json.loads(monitor_path.read_text())
    if not data:
        raise SystemExit("RMSE monitor file is empty")
    budgets = sorted({entry.get("stage") for entry in data if entry.get("stage") is not None})
    fig, ax = plt.subplots()
    for stage in budgets:
        rmse_vals = [entry["combined_rmse"] for entry in data if entry.get("stage") == stage]
        if not rmse_vals:
            continue
        ax.plot(range(len(rmse_vals)), rmse_vals, label=f"stage {stage}")
    ax.set_title("Combined RMSE progression")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Combined RMSE")
    ax.legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    print(f"Saved RMSE progression plot to {plot_path}")


if __name__ == "__main__":
    main()
