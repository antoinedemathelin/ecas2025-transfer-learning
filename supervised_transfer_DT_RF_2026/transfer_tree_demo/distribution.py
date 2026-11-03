"""Store and sample the exact seven-component Gaussian mixture used in TransferLab.

Classification labels are deterministic conditional on the latent component C.
Regression uses component-specific baseline, local slope, nonlinear effect and noise.

The JSON contains the distribution parameters only; it does not contain a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def covariance_from_parameters(major: float, minor: float, angle_deg: float) -> np.ndarray:
    """Covariance with principal standard deviations major/minor and rotation angle."""
    angle = np.deg2rad(angle_deg)
    ca, sa = np.cos(angle), np.sin(angle)
    rotation = np.array([[ca, -sa], [sa, ca]])
    return rotation @ np.diag([major**2, minor**2]) @ rotation.T


def validate_distribution(parameters: dict[str, Any]) -> None:
    """Check the schema and dimensional consistency before generating data."""
    if parameters.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION}")

    for field in (
        "weights", "means", "cov_params", "labels", "regression_base_levels",
        "regression_local_amplitudes", "regression_gradient_angles_deg",
        "regression_noise_std", "regression_curvature_amplitude",
        "regression_curvature_frequency",
    ):
        if field not in parameters:
            raise ValueError(f"Missing distribution parameter: {field}")

    weights = np.asarray(parameters["weights"], dtype=float)
    means = np.asarray(parameters["means"], dtype=float)
    cov_params = np.asarray(parameters["cov_params"], dtype=float)
    labels = np.asarray(parameters["labels"])
    k = len(weights)
    if k < 1 or weights.ndim != 1 or means.shape != (k, 2) or cov_params.shape != (k, 3):
        raise ValueError("Expected weights (K,), means (K,2), cov_params (K,3)")
    if labels.shape != (k,) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Classification labels must be 0/1 with one label per cluster")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Mixture weights must be nonnegative and sum to 1")
    if not np.isfinite(means).all() or not np.isfinite(cov_params).all() or np.any(cov_params[:, :2] <= 0):
        raise ValueError("Means must be finite and major/minor standard deviations positive")
    for field in ("regression_base_levels", "regression_local_amplitudes", "regression_gradient_angles_deg"):
        array = np.asarray(parameters[field], dtype=float)
        if array.shape != (k,) or not np.isfinite(array).all():
            raise ValueError(f"{field} must be a finite array of length K")
    for field in ("regression_noise_std", "regression_curvature_amplitude", "regression_curvature_frequency"):
        value = float(parameters[field])
        if not np.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if float(parameters["regression_noise_std"]) < 0:
        raise ValueError("regression_noise_std must be nonnegative")


def load_distribution(path: str | Path) -> dict[str, Any]:
    """Load and validate the distribution specification from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        parameters = json.load(f)
    validate_distribution(parameters)
    return parameters


def save_distribution(parameters: dict[str, Any], path: str | Path) -> None:
    """Save a validated distribution specification as readable JSON."""
    validate_distribution(parameters)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2, allow_nan=False)
        f.write("\n")


def sample_from_distribution(
    parameters: dict[str, Any], n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw fresh (X, class_labels, regression_values, hidden_cluster_ids).

    A new seed yields an independent sample from the *same* joint distribution.
    Passing the same seed yields identical draws with the same NumPy RNG version.
    """
    validate_distribution(parameters)
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 0:
        raise ValueError("n must be a nonnegative integer")

    rng = np.random.default_rng(seed)
    weights = np.asarray(parameters["weights"], dtype=float)
    means = np.asarray(parameters["means"], dtype=float)
    labels = np.asarray(parameters["labels"], dtype=int)
    covariances = np.array([covariance_from_parameters(*v) for v in parameters["cov_params"]])
    base = np.asarray(parameters["regression_base_levels"], dtype=float)
    amplitudes = np.asarray(parameters["regression_local_amplitudes"], dtype=float)
    angles = np.deg2rad(np.asarray(parameters["regression_gradient_angles_deg"], dtype=float))
    slopes = amplitudes[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    noise_std = float(parameters["regression_noise_std"])
    curve_amp = float(parameters["regression_curvature_amplitude"])
    curve_freq = float(parameters["regression_curvature_frequency"])

    clusters = rng.choice(len(weights), size=n, p=weights)
    X = np.empty((n, 2), dtype=float)
    regression = np.empty(n, dtype=float)
    for k in range(len(weights)):
        indices = np.flatnonzero(clusters == k)
        if len(indices) == 0:
            continue
        X[indices] = rng.multivariate_normal(means[k], covariances[k], size=len(indices))

    classification = labels[clusters]  # No label noise: Y_class = label[C].
    for k in range(len(weights)):
        indices = np.flatnonzero(clusters == k)
        if len(indices) == 0:
            continue
        delta = X[indices] - means[k]
        curved = curve_amp * np.sin(curve_freq * delta[:, 0] * delta[:, 1])
        regression[indices] = (
            base[k]
            + delta @ slopes[k]
            + curved
            + rng.normal(0, noise_std, size=len(indices))
        )

    return X, classification, regression, clusters


def sample_from_json(
    path: str | Path, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-line entry point to generate data from a saved distribution."""
    return sample_from_distribution(load_distribution(path), n, seed)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    path = root / "source_distribution_7clusters.json"
    X, y_class, y_reg, cluster = sample_from_json(path, n=2000, seed=42)
    print("X:", X.shape, "classification:", y_class.shape, "regression:", y_reg.shape)
    print("Counts by component:", np.bincount(cluster, minlength=7))
