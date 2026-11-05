"""Run the classification and/or regression reproduction from any directory."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

# Keep small-data experiments from launching large BLAS thread pools.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "classification", "regression"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "Historical classification/regression reproduction; see README for evaluation limitations.",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "openpyxl", "catboost")},
        "tasks": {},
    }
    if args.task in ("all", "classification"):
        from reproduction.classification import run
        manifest["tasks"]["classification"] = run(destination)
    if args.task in ("all", "regression"):
        from reproduction.regression import run
        manifest["tasks"]["regression"] = run(destination)
    manifest_path = destination / f"manifest_{args.task}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults: {destination}\nRun manifest: {manifest_path}")


if __name__ == "__main__":
    main()
