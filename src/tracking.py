"""MLflow experiment tracking helpers.

# TODO(owner: B) — implement log_run().
"""

# TODO(owner: B)

from __future__ import annotations

from pathlib import Path


def log_run(
    config: dict,
    metrics: dict[str, float],
    artifacts: list[Path] | None = None,
) -> str:
    """Log a training run to MLflow.

    Logs the git commit, data version, config, seed, and torch/Python versions
    automatically alongside the provided metrics.

    Parameters
    ----------
    config:
        Parsed YAML config dictionary.
    metrics:
        Dictionary of metric names to scalar values, e.g.
        ``{"val_loss": 1.23, "val_perplexity": 3.42}``.
    artifacts:
        Optional list of file paths to log as MLflow artefacts.

    Returns
    -------
    str
        The MLflow run ID.

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)
