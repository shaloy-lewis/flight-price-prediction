"""Versioned local artifacts: verify integrity/compatibility before trusted deserialization."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib

from src.components.data_ingestion import sha256_file
from src.reporting import write_json
from src.utils.constants import FEATURES, SCHEMA_VERSION

COMPATIBLE_PACKAGES = ("scikit-learn", "numpy", "pandas", "scipy", "joblib")


class ArtifactError(RuntimeError):
    pass


def runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in COMPATIBLE_PACKAGES}


def save_artifact(directory: Path, estimator, metadata: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ArtifactError("Artifact already exists; use a new versioned directory")
    path = directory / "model.joblib"
    joblib.dump(estimator, path, compress=3)
    manifest = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "features": FEATURES,
        "python": ".".join(platform.python_version_tuple()[:2]),
        "packages": runtime_versions(),
        "model_sha256": sha256_file(path),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_artifact(directory: Path | str, *, trusted: bool = False):
    """A hash detects corruption, not malicious models. Only load a trusted local run."""
    if not trusted:
        raise ArtifactError("Only explicitly trusted, locally controlled artifacts may be loaded")
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION or manifest["features"] != FEATURES:
            raise ArtifactError("Incompatible model input schema; retrain with the current code")
        if manifest["packages"] != runtime_versions():
            raise ArtifactError("Dependency mismatch; install the run's locked environment")
        if manifest["python"] != ".".join(platform.python_version_tuple()[:2]):
            raise ArtifactError("Python minor version mismatch")
        path = directory / "model.joblib"
        if sha256_file(path) != manifest["model_sha256"]:
            raise ArtifactError("Model integrity check failed")
        for field in (
            "model_version",
            "selected_model",
            "reference",
            "target_units",
            "data_sources",
        ):
            if field not in manifest:
                raise ArtifactError(f"Missing model metadata: {field}")
        return joblib.load(path), manifest
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("Cannot load the versioned model artifact") from exc
