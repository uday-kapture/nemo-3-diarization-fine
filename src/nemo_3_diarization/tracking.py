"""MLflow for training and evaluation (as in nemotron-3.5-asr).

Traps this avoids:
- with no tracking URI, runs land silently in a local store on whichever box ran them, so a missing
  MLFLOW_TRACKING_URI is an error;
- with both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_TOKEN set, MLflow sends Basic auth only.

Training logs through Lightning's MLFlowLogger and writes its run id to <out-dir>/mlflow_run_id, so evaluating that
run's .nemo nests the evaluation under it. Weights are never logged (log_model=False).
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

PARENT_TAG = "mlflow.parentRunId"
RUN_ID_FILE = "mlflow_run_id"


class TrackingError(RuntimeError):
    pass


def tracking_uri() -> str:
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_TOKEN"):
        print("warning: both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_TOKEN are set; "
              "MLflow will use Basic auth only", file=sys.stderr)
    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        return uri
    raise TrackingError("MLFLOW_TRACKING_URI is not set; runs would land on this box only. "
                        "Fill in .env from .env.example and `source .env`.")


def numeric(metrics: dict, prefix: str = "") -> dict[str, float]:
    """The finite numbers in `metrics` (None, NaN and non-numbers dropped), prefixed."""
    return {f"{prefix}{k}": float(v) for k, v in metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool) and v == v}


@contextmanager
def run(experiment: str, run_name: str, *, stage: str, parent_run_id: str | None = None,
        tags: dict | None = None) -> Iterator:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({"stage": stage, **({PARENT_TAG: parent_run_id} if parent_run_id else {}), **(tags or {})})
        yield active


def lightning_logger(experiment: str, run_name: str, *, tags: dict | None = None):
    from lightning.pytorch.loggers import MLFlowLogger

    return MLFlowLogger(experiment_name=experiment, run_name=run_name, tracking_uri=tracking_uri(), tags=tags,
                        log_model=False)
