"""Single command-line entry point for reproducible offline workflows."""

import argparse
import logging
from pathlib import Path

from src.components.data_ingestion import load_dataset
from src.config import ExperimentConfig
from src.reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical fare benchmark workflows")
    parser.add_argument("command", choices=["validate", "eda", "train", "evaluate", "drift"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/local"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = ExperimentConfig.load(args.config)
    if args.command == "validate":
        dataset = load_dataset(args.data_dir)
        write_json(args.output / "data_quality.json", dataset.quality)
        write_json(args.output / "sources.json", dataset.sources)
        print(f"Retained {len(dataset.frame):,} valid offers")
    elif args.command == "eda":
        from src.analysis import run_analysis

        run_analysis(args.data_dir, args.output, config)
    elif args.command == "train":
        from src.pipeline.training_pipeline import run_training

        run_training(args.data_dir, args.output, config)
    elif args.command == "evaluate":
        import numpy as np
        import pandas as pd

        from src.artifacts import load_artifact
        from src.components.data_ingestion import sha256_file
        from src.components.model_evaluation import regression_metrics

        model, manifest = load_artifact(args.model_dir, trusted=True)
        dataset = load_dataset(args.data_dir)
        if dataset.sources != manifest["data_sources"]:
            raise ValueError("Dataset differs from the locked experiment")
        split = pd.read_csv(args.model_dir / "split_assignments.csv")
        if (
            sha256_file(args.model_dir / "split_assignments.csv")
            != manifest["evaluation"]["split_sha256"]
        ):
            raise ValueError("Split assignment integrity check failed")
        holdout = dataset.frame.iloc[split.loc[split.partition == "holdout", "row"]]
        from src.utils.constants import FEATURES

        prediction = np.asarray(model.predict(holdout[FEATURES]))
        write_json(
            args.output / "recomputed_metrics.json",
            regression_metrics(holdout.price.to_numpy(), prediction),
        )
    else:
        from src.artifacts import load_artifact
        from src.monitoring import drift_report

        _, manifest = load_artifact(args.model_dir, trusted=True)
        frame = load_dataset(args.data_dir).frame
        write_json(args.output / "drift.json", drift_report(frame, manifest["reference"]))


if __name__ == "__main__":
    main()
