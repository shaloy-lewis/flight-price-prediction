"""Explicit experiment configuration; no environment-dependent training defaults."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    holdout_fraction: float = 0.2
    cv_folds: int = 3
    bootstrap_repetitions: int = 500
    permutation_sample: int = 2000
    permutation_repeats: int = 5
    models: tuple[str, ...] = (
        "median",
        "route_class_median",
        "ridge",
        "tree",
        "tree_log",
        "hist",
        "hist_log",
    )

    def __post_init__(self) -> None:
        if not 0 < self.holdout_fraction < 0.5 or self.cv_folds < 2:
            raise ValueError("Require a holdout fraction in (0, .5) and at least two CV folds")
        if min(self.bootstrap_repetitions, self.permutation_sample, self.permutation_repeats) < 1:
            raise ValueError("Evaluation sample sizes must be positive")
        allowed = {"median", "route_class_median", "ridge", "tree", "tree_log", "hist", "hist_log"}
        if (
            not self.models
            or len(set(self.models)) != len(self.models)
            or set(self.models) - allowed
        ):
            raise ValueError("Models must be a nonempty unique list of supported candidates")
        if not {"median", "route_class_median", "ridge", "tree"}.issubset(self.models):
            raise ValueError(
                "Include the required statistical, route/class, linear and tree baselines"
            )

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "models" in values:
            values["models"] = tuple(values["models"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
