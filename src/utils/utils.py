from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import mlflow
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    import numpy as np
    from mlflow.entities import Run
    from omegaconf import DictConfig


def already_run(cfg: DictConfig, experiment_name: str) -> bool:
    """Check if it was already runned.

    Returns:
        bool
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)  # get all experiments
    if experiment is not None:
        runs = client.search_runs(
            [experiment.experiment_id],
        )  # get all runs from an experiment
        for run in runs:
            if (
                run.data.params == {k: str(v) for k, v in cfg.items()}
                and "error_message" not in run.data
            ):
                return True
    return False


def values_as_list(d: DictConfig, exclude_k: str = "_target_") -> list[str]:
    """DictConfig as string excluding a field.

    Args:
        d: DictConfig to convert
        exclude_k: string to exclude

    Returns:
        list of string with keys and values.
    """
    return [f"{k}{v}" for k, v in d.items() if k != exclude_k]


def get_metrics_from_values(
    acc_origins: np.array, acc_frauds: np.array,
) -> dict[str, float]:
    """Compute metrics from two lists.

    Returns:
        dict with metrics
    """
    # acc_frauds is what was predicted, 1 means that its a tp
    # calculating stats
    origin_fp = acc_origins
    frauds_tp = acc_frauds

    tp = sum(frauds_tp)
    fp = sum(origin_fp)
    fn = len(frauds_tp) - tp
    tn = len(origin_fp) - fp
    if len(acc_frauds) == 0:  # only origins
        return {
            "specificity": 1 - sum(acc_origins) / len(acc_origins),
            "len": len(acc_origins),
        }
    if len(acc_origins) == 0:  # only frauds
        return {"recall": sum(acc_frauds) / len(acc_frauds), "len": len(acc_frauds)}

    precision = 0
    recall = 0
    fscore = 0

    sumof = sum(acc_origins) + sum(acc_frauds)
    if sumof:
        precision = sum(acc_frauds) / sumof
        recall = sum(acc_frauds) / len(acc_frauds)
        if not precision or not recall:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)

    return {
        "fscore": fscore,
        "recall": recall,
        "precision": precision,
        "specificity": 1 - sum(acc_origins) / len(acc_origins),
        "fp": fp,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "len": len(acc_origins) + len(acc_frauds),
    }


def get_float32():
    return torch.float32


def get_metrics(dataset, model, decision) -> dict:
    """Get metrics for a dataset model and decision method.

    Returns:
        dict of metrics
    """
    # the predictions
    acc_frauds = []
    acc_origins = []

    for i in tqdm(range(len(dataset))):
        model.reset()
        frames = dataset[i]
        d, _ = decision.process_frame_by_frame(frames, model)
        if dataset.isFraud(i):
            acc_frauds.append(d)
        else:
            acc_origins.append(d)

    return get_metrics_from_values(acc_origins, acc_frauds)


def get_fscore(thr: float, origins_t: np.array, frauds_t: np.array) -> dict:
    """Compute the fscore.

    Returns:
        fscore and other metrics
    """
    origin_fp = origins_t < thr
    frauds_tp = frauds_t < thr

    return get_metrics_from_values(origin_fp, frauds_tp)["fscore"]


def mlruntodict(mlparams: str) -> dict:
    """Convert mlrun string to dict.

    Returns:
        config dict
    """
    return {k: (ast.literal_eval(v) if "'" in v else v) for k, v in mlparams.items()}


def get_best_run(
    experiment_name: str,
    metrics_name: str,
    task_name: str | None = None,
    run_name: str | None = None,
    params: dict | None = None,
) -> Run | None:
    """Get the best run from experiments.

    Returns:
        best run if founded else None
    """
    # retrieves the runs from the experiments
    client = mlflow.MlflowClient()
    current_experiment = dict(client.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment["experiment_id"]
    runs = client.search_runs([experiment_id], order_by=[f"metrics.{metrics_name} ASC"])

    if params is None and task_name is None:
        return runs[0]

    if task_name is not None:
        for run in runs:
            if task_name == run.data.tags.get("task_name", "") and (
                run_name is not None and run.info.run_name == run_name
            ):
                return run
    if params is not None:
        for run in runs:
            if run.data.params == {k: str(v) for k, v in params.items()}:
                return run
    return None
