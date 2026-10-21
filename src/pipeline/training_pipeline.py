"""Reproducible grouped model development and a locked final evaluation.

No data is loaded at import time. Run through python -m src.cli train.
"""

import logging
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold
from threadpoolctl import threadpool_limits

from src.artifacts import save_artifact
from src.components.data_ingestion import holdout_split, leakage_groups, load_dataset, sha256_file
from src.components.model_evaluation import cluster_bootstrap, regression_metrics, segment_metrics
from src.components.model_trainer import candidate, fitted_pipeline, target_units
from src.config import ExperimentConfig
from src.explainability import Explainer
from src.monitoring import make_reference
from src.reporting import error_plots, write_json
from src.utils.constants import FEATURES, PROJECT_VERSION, SCOPE

logger = logging.getLogger(__name__)


def run_training(data_dir: Path, output: Path, config: ExperimentConfig) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory must be empty; use a new versioned run to preserve evidence"
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(data_dir)
    frame = dataset.frame
    groups = leakage_groups(frame)
    train_idx, test_idx = holdout_split(frame, groups, config.seed, config.holdout_fraction)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    X, y = train[FEATURES], train.price.to_numpy(dtype=float)
    if len(np.unique(groups[train_idx])) < config.cv_folds:
        raise ValueError("Insufficient independent training groups for requested CV")
    splits = list(GroupKFold(config.cv_folds).split(X, y, groups[train_idx]))
    comparisons, models = [], {}
    with threadpool_limits(limits=1):
        for name in config.models:
            logger.info("Developing %s on %s training offers", name, len(train))
            estimator, grid = candidate(name, config.seed)
            start = time.perf_counter()
            search = GridSearchCV(
                estimator,
                grid,
                cv=splits,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X, y)
            elapsed = time.perf_counter() - start
            best = search.best_index_
            folds = [
                -float(search.cv_results_[f"split{idx}_test_score"][best])
                for idx in range(config.cv_folds)
            ]
            comparisons.append(
                {
                    "model": name,
                    "cv_mae_inr": -float(search.best_score_),
                    "cv_fold_mae_inr": folds,
                    "cv_std_mae_inr": float(search.cv_results_["std_test_score"][best]),
                    "fit_search_seconds": elapsed,
                    "best_parameters": search.best_params_,
                }
            )
            models[name] = search.best_estimator_
            write_json(output / "cv_progress.json", comparisons)
        # Selection is frozen before ANY holdout prediction. Ties use the prescribed model order.
        selected = min(
            comparisons,
            key=lambda row: (row["cv_mae_inr"], list(config.models).index(row["model"])),
        )["model"]
        selection = {
            "selected_model": selected,
            "rule": "minimum mean grouped-CV MAE in original INR; fixed order breaks exact ties",
            "holdout_used_for_selection": False,
            "comparisons": comparisons,
        }
        write_json(output / "selection.json", selection)
        predictions = {}
        for result in comparisons:
            name = result["model"]
            predictions[name] = np.asarray(models[name].predict(test[FEATURES]), dtype=float)
            result["holdout"] = regression_metrics(test.price.to_numpy(), predictions[name])
        winner = models[selected]
        prediction = predictions[selected]
        uncertainty = cluster_bootstrap(
            test.price,
            prediction,
            groups[test_idx],
            config.bootstrap_repetitions,
            config.seed,
            predictions["route_class_median"],
        )
        edges = np.quantile(y, [0.25, 0.5, 0.75])
        if len(np.unique(edges)) < 3:
            edges = np.array([2500.0, 10000.0, 30000.0])
        segments = segment_metrics(test, prediction, edges)
        segments.to_csv(output / "segment_errors.csv", index=False)
        errors = test[FEATURES + ["flight_id", "travel_date", "price"]].copy()
        errors["prediction_inr"] = prediction
        errors["absolute_error_inr"] = np.abs(test.price.to_numpy() - prediction)
        errors["group"] = groups[test_idx]
        errors.nlargest(25, "absolute_error_inr").to_csv(output / "largest_errors.csv", index=False)
        errors.to_csv(output / "holdout_predictions.csv", index=False)
        error_plots(test.price.to_numpy(), prediction, output / "figures", config.seed)
        sample = test.sample(min(config.permutation_sample, len(test)), random_state=config.seed)
        importance = permutation_importance(
            winner,
            sample[FEATURES],
            sample.price,
            scoring="neg_mean_absolute_error",
            n_repeats=config.permutation_repeats,
            random_state=config.seed,
            n_jobs=1,
        )
        permutation = [
            {"feature": feature, "mae_increase_inr": float(mean), "repeat_std_inr": float(std)}
            for feature, mean, std in zip(
                FEATURES, importance.importances_mean, importance.importances_std
            )
        ]
        processed_mean = None
        pipeline = fitted_pipeline(winner)
        if hasattr(pipeline, "named_steps"):
            processed_mean = pipeline.named_steps["preprocessor"].transform(X).mean(axis=0).tolist()
        explainer = Explainer(winner, processed_mean)
        local = explainer.explain(sample[FEATURES].iloc[:1])
        # Small descriptive sample, computed offline. These are encoded-feature attributions.
        attributions = [
            explainer.explain(sample[FEATURES].iloc[idx : idx + 1])
            for idx in range(min(100, len(sample)))
        ]
        keys = list(local["contributions"])
        global_attribution = [
            {
                "feature": key,
                "mean_absolute_contribution": float(
                    np.mean([abs(row["contributions"][key]) for row in attributions])
                ),
            }
            for key in keys
        ]
        write_json(
            output / "explanations.json",
            {
                "local_example": local,
                "sample_size": len(attributions),
                "units": local["units"],
                "method": local["method"],
                "mean_absolute_attribution": global_attribution,
            },
        )
        write_json(output / "permutation_importance.json", permutation)
    split_table = pd.DataFrame(
        {"row": np.arange(len(frame)), "group": groups, "partition": "train"}
    )
    split_table.loc[test_idx, "partition"] = "holdout"
    split_table.to_csv(output / "split_assignments.csv", index=False)
    split_hash = sha256_file(output / "split_assignments.csv")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    source_files = sorted(Path("src").rglob("*.py")) + [
        Path("api.py"),
        Path("app.py"),
        Path("pyproject.toml"),
        Path("configs/experiment.json"),
    ]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for source in source_files:
            if source.is_file():
                snapshot.write(source, source.as_posix())
    source_hash = sha256_file(output / "source_snapshot.zip")
    summary = {
        "source_snapshot_sha256": source_hash,
        "selected_model": selected,
        "selection_rule": selection["rule"],
        "comparisons": comparisons,
        "uncertainty": uncertainty,
        "train_rows": len(train),
        "holdout_rows": len(test),
        "train_groups": len(np.unique(groups[train_idx])),
        "holdout_groups": len(np.unique(groups[test_idx])),
        "total_groups": len(np.unique(groups)),
        "largest_group_rows": int(pd.Series(groups).value_counts().max()),
        "holdout_row_fraction": len(test) / len(frame),
        "split_sha256": split_hash,
        "price_bucket_edges_inr_from_training": edges.tolist(),
        "config": config.to_dict(),
        "quality": dataset.quality,
        "data_sources": dataset.sources,
        "revision": revision,
        "working_tree_dirty": dirty,
    }
    write_json(output / "results.json", summary)
    write_json(output / "config.json", config.to_dict())
    environment = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata["Name"]
    )
    (output / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    metadata = {
        "model_version": output.name,
        "selected_model": selected,
        "project_version": PROJECT_VERSION,
        "scope": SCOPE,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_units": target_units(winner),
        "prediction_units": "INR",
        "config": config.to_dict(),
        "reference": make_reference(X),
        "processed_training_mean": processed_mean,
        "data_sources": dataset.sources,
        "evaluation": summary,
        "permutation_importance": permutation,
        "revision": revision,
        "working_tree_dirty": dirty,
    }
    save_artifact(output, winner, metadata)
    logger.info(
        "Selected %s; holdout MAE INR %.2f",
        selected,
        regression_metrics(test.price.to_numpy(), prediction)["mae_inr"],
    )
    return summary
