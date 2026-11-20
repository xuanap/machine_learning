"""Run A's diagnostic or prepared-input clustering experiments."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.inputs import ROOT, load_input
from experiments.runner import run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--suite", choices=["baseline", "parameters", "ablation", "all"], default="baseline")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample-size", type=int, default=300, help="Only used in smoke mode")
    parser.add_argument("--output", type=Path, help="A new, non-existing output directory")
    args = parser.parse_args()
    try:
        bundle = load_input(args.mode, args.input, args.metadata, args.sample_size)
        config_path = args.config or ROOT / "config" / ("smoke_experiments.json" if args.mode == "smoke" else "experiment_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or ROOT / "results" / args.mode / f"{args.suite}_{timestamp}"
        table = run_suite(bundle, config, output, args.suite)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Mode: {args.mode}; runs: {len(table)}; status counts: {table.status.value_counts().to_dict()}")
    print(f"Saved to: {output.resolve()}")
    if args.mode == "smoke":
        print("Diagnostic run only: not a formal clustering conclusion.")
    if table.status.eq("failed").any() or table.status.eq("metric_error").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
