#!/usr/bin/env python3
"""Run the latest drawing utilities without memorizing each entry point."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent

Builder = Callable[[argparse.Namespace], List[str]]


class ScriptSpec:
    def __init__(self, name: str, builder: Builder) -> None:
        self.name = name
        self.path = SCRIPT_DIR / name
        self.builder = builder

    def command(self, cfg: argparse.Namespace) -> List[str]:
        return self.builder(cfg)


def _build_search_results(cfg: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "plot_search_results.py"),
        "--output-dir",
        str(cfg.search_dir),
    ]


def _build_model_vs_gt(cfg: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_gt_vs_sim.py"),
        "--search-dir",
        str(cfg.search_dir),
        "--gt-dir",
        str(cfg.gt_dir),
        "--out",
        str(cfg.model_vs_gt_out),
    ]
    if cfg.month is not None:
        cmd.extend(["--month", str(cfg.month)])
    return cmd


def _build_param_evolution(cfg: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "plot_param_evolution.py"),
        "--search-dir",
        str(cfg.search_dir),
        "--out-dir",
        str(cfg.plots_dir),
    ]


def _build_all_daily(cfg: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "plot_all_daily.py"),
        "--search-dir",
        str(cfg.search_dir),
        "--workers",
        str(cfg.workers),
    ]


def _build_plot_viewer(cfg: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "build_plot_viewer.py"),
        "--plots-dir",
        str(cfg.plots_dir),
    ]


def _build_temporal_three_panels(cfg: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "plot_temporal_params_three_panels.py"),
        "--search-dir",
        str(cfg.search_dir),
    ]


ALL_SCRIPTS: Sequence[ScriptSpec] = [
    ScriptSpec("plot_search_results.py", _build_search_results),
    ScriptSpec("plot_gt_vs_sim.py", _build_model_vs_gt),
    ScriptSpec("plot_param_evolution.py", _build_param_evolution),
    ScriptSpec("plot_all_daily.py", _build_all_daily),
    ScriptSpec("plot_temporal_params_three_panels.py", _build_temporal_three_panels),
    ScriptSpec("build_plot_viewer.py", _build_plot_viewer),
]


def _select_scripts(
    configs: Sequence[ScriptSpec], args: argparse.Namespace
) -> List[ScriptSpec]:
    ordered = sorted(configs, key=lambda spec: spec.path.stat().st_mtime, reverse=True)
    if args.scripts:
        requested: Set[str] = {Path(name).name for name in args.scripts}
        selected = [spec for spec in ordered if spec.name in requested]
        missing = requested - {spec.name for spec in selected}
        if missing:
            raise SystemExit(f"Unknown scripts requested: {', '.join(sorted(missing))}")
        return selected
    if args.limit == 0:
        return ordered
    return ordered[: args.limit]


def _run_script(spec: ScriptSpec, cfg: argparse.Namespace, dry_run: bool) -> None:
    if not spec.path.exists():
        raise SystemExit(f"Script {spec.name} not found in {SCRIPT_DIR}")
    cmd = spec.command(cfg)
    print(f"\n==> Running {spec.name}")
    print("    " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the most recently updated drawing scripts in order."
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        required=True,
        help="Search output directory used by the plotting scripts.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory for generated plots (defaults to search_dir/plots).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Ground truth directory (defaults to search_dir/../gt).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of most recently modified scripts to run (0 = all).",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        help="Explicit script filenames to run instead of using --limit.",
    )
    parser.add_argument(
        "--model-vs-gt-out",
        type=Path,
        default=None,
        help="Path where plot_gt_vs_sim.py should write its output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers passed to plot_all_daily.py.",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Month argument forwarded to plot_gt_vs_sim.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the commands without executing them.",
    )

    args = parser.parse_args()
    search_dir = args.search_dir.resolve()
    if not search_dir.exists():
        raise SystemExit(f"{search_dir} does not exist")
    plots_dir = (args.plots_dir or search_dir / "plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = (args.gt_dir or search_dir.parent / "gt").resolve()
    model_vs_gt_out = (args.model_vs_gt_out or plots_dir / "model_vs_gt.png").resolve()

    cfg = argparse.Namespace(
        search_dir=search_dir,
        plots_dir=plots_dir,
        gt_dir=gt_dir,
        workers=args.workers,
        month=args.month,
        model_vs_gt_out=model_vs_gt_out,
    )

    selected = _select_scripts(ALL_SCRIPTS, args)
    if not selected:
        raise SystemExit("No drawing scripts selected for execution.")

    print("Scripts to run:")
    for spec in selected:
        print(f"  - {spec.name} (modified: {spec.path.stat().st_mtime:.0f})")

    for spec in selected:
        _run_script(spec, cfg, args.dry_run)

    print("\nAll requested drawing scripts finished.")


if __name__ == "__main__":
    main()
