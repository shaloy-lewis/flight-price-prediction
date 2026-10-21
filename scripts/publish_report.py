"""Copy small executed evidence into a reviewable directory; never publish the model or data."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/reference"))
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "results.json",
        "selection.json",
        "config.json",
        "segment_errors.csv",
        "largest_errors.csv",
        "permutation_importance.json",
        "explanations.json",
    ):
        shutil.copy2(args.run / name, args.output / name)
    shutil.copytree(args.run / "figures", args.output / "figures", dirs_exist_ok=True)
    comparisons = pd.DataFrame(
        [
            {
                "model": row["model"],
                "cv_mae_inr": row["cv_mae_inr"],
                "cv_std_mae_inr": row["cv_std_mae_inr"],
                "fit_search_seconds": row["fit_search_seconds"],
                **row["holdout"],
            }
            for row in result["comparisons"]
        ]
    )
    comparisons.to_csv(args.output / "model_comparison.csv", index=False)
    # Environment is text and small. Binary model, full holdout predictions, and source zip stay local.
    shutil.copy2(args.run / "environment.txt", args.output / "environment.txt")
    print(f"Published actual run evidence for {result['selected_model']}")


if __name__ == "__main__":
    main()
