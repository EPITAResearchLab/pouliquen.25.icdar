# A logger for this file
import json
import logging
import os
import random
import shutil
import tempfile
import uuid

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

from src.utils.utils import (
    already_run,
    get_metrics,
    get_metrics_from_values,
    mlruntodict,
)

log = logging.getLogger(__name__)

def get_raw_predictions(dataset, model, decision):
    """Get raw prediction scores and true labels for a dataset.

    Args:
        dataset: The dataset to process
        model: Model for processing
        decision: Decision method

    Returns:
        tuple: (y_true, y_score) arrays
    """
    # Get all predictions with true labels
    y_true = []
    y_score = []

    for i in tqdm(range(len(dataset))):
        model.reset()
        frames = dataset[i]

        # Get raw score
        raw_score, _ = decision.process(frames, model)

        # Record true label (1 for fraud, 0 for legitimate)
        is_fraud = dataset.isFraud(i)
        y_true.append(1 if is_fraud else 0)

        y_score.append(1 - raw_score)

    # Convert to numpy arrays
    return np.array(y_true), np.array(y_score)

def merge_metrics(metrics_list, dataset_sizes):
    """Merge metrics from multiple datasets, weighting by dataset size.

    Each sample has equal weight.

    Args:
        metrics_list: List of metrics dictionaries to merge
        dataset_sizes: List of dataset sizes

    Returns:
        Dictionary with merged metrics weighted by dataset size
    """
    if not metrics_list or not dataset_sizes or len(metrics_list) != len(dataset_sizes):
        if not metrics_list:
            return {}
        # Fallback to equal weighting if sizes not provided properly
        return {key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
               for key in set().union(*[set(m.keys()) for m in metrics_list])}

    # Initialize result with all keys from all metrics
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    merged = {}
    total_samples = sum(dataset_sizes)

    # For each key, compute the weighted average based on dataset sizes
    for key in all_keys:
        weighted_sum = 0
        weighted_count = 0

        for i, metrics in enumerate(metrics_list):
            if key in metrics:
                dataset_weight = dataset_sizes[i] / total_samples
                weighted_sum += metrics[key] * dataset_weight
                weighted_count += dataset_weight

        if weighted_count > 0:
            merged[key] = weighted_sum

    return merged


def pipeline_with_roc(cfg, params, task_name_full, log, dataset_groups=None):
    """Run the pipeline with ROC curve support and threshold comparison.

    Args:
        cfg: Configuration object (hydra like)
        params: Model parameters
        task_name_full: task name for MLflow
        log: Logger object
        dataset_groups: Optional dictionary mapping group names to lists of dataset names
                        that should be merged.
                        Example: {"swap": ["swap", "swapthree"]}

    The first dataset is assumed to have both legit and fraud samples,
    other datasets are assumed to be fraud-only.
    """
    run_uuid = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"mlflow_run_{run_uuid}")
    os.makedirs(temp_dir, exist_ok=True)

    log.info(f"Created temporary directory for this run: {temp_dir}")
    standard_thresholds = np.linspace(0, 1, 100)  # 100 evenly spaced points
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log.info(f"Started MLflow run with ID: {run_id}")

            mlflow.set_tag("mlflow.runName", f"{task_name_full}")
            mlflow.log_params(params)

            # Dictionary to store all metrics across datasets
            all_metrics = {}
            all_roc_data = {}
            all_raw_scores = {}
            dataset_sample_sizes = {}

            # Get list of test datasets
            test_datasets = list(cfg.data.test.keys())

            # First dataset is the main one with both classes
            main_dataset = test_datasets[0] if test_datasets else None

            # Process each test dataset
            for i, test_d in enumerate(test_datasets):
                seed_everything(cfg.seed, workers=True)
                model = instantiate(params.model)
                decision = instantiate(params.decision)
                data_test = instantiate(cfg.data.test[test_d])

                log.info("Processing dataset %s on path %s",
                       test_d, str(data_test.input_dir))

                # Get standard metrics
                metrics = get_metrics(data_test, model, decision)
                all_metrics[test_d] = metrics

                # Store the dataset size
                dataset_sample_sizes[test_d] = len(data_test)

                # Log standard metrics to MLflow
                mlflow.log_metrics({f"{test_d}_{k}": metrics[k] for k in metrics})

                # For all datasets, collect raw scores for threshold comparison
                y_true, y_score = get_raw_predictions(data_test, model, decision)
                all_raw_scores[test_d] = {
                    "y_true": y_true,
                    "y_score": y_score,
                }

                # First dataset has both classes, calculate ROC
                if i == 0:
                    # # Create unique file path in the temp directory
                    roc_data_path = os.path.join(temp_dir, f"{test_d}_roc_data.json")

                    fpr, tpr, orig_thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
                    print("num thresholds  ", len(orig_thresholds))
                    roc_auc = auc(fpr, tpr)

                    sorted_indices = np.argsort(orig_thresholds)  # Sort in ascending order
                    sorted_thresholds = orig_thresholds[sorted_indices]
                    sorted_fpr = fpr[sorted_indices]
                    sorted_tpr = tpr[sorted_indices]

                    # Interpolate values to fixed threshold grid
                    interpolated_fpr = np.interp(standard_thresholds, sorted_thresholds, sorted_fpr)
                    interpolated_tpr = np.interp(standard_thresholds, sorted_thresholds, sorted_tpr)

                    all_roc_data[test_d] = {
                        "fpr": interpolated_fpr,
                        "tpr": interpolated_tpr,
                        "thresholds": standard_thresholds,
                        "auc": roc_auc,
                    }

                    # Store standardized values
                    with open(roc_data_path, "w") as f:
                        json.dump({
                            "fpr": interpolated_fpr.tolist(),
                            "tpr": interpolated_tpr.tolist(),
                            "thresholds": standard_thresholds.tolist(),
                            "auc": float(roc_auc),
                            "original_data": {  # Keep original for ref.
                                "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                                "thresholds": orig_thresholds.tolist()
                            },
                            "num_samples": len(y_true),
                            "seed": cfg.seed,
                        }, f)
                    mlflow.log_metric(f"{test_d}_auc", roc_auc)
                    mlflow.log_artifact(roc_data_path, "roc_data")

                else:
                    # For fraud-only datasets, calculate recalls at ALL thresholds from main dataset
                    print("fraudddd only, logging recall")
                    print(main_dataset)
                    if main_dataset and main_dataset in all_roc_data:
                        threshold_data_path = os.path.join(temp_dir, f"{test_d}_threshold_data.json")

                        recalls = []
                        for threshold in standard_thresholds:
                            y_pred = y_score >= threshold
                            recall = np.sum(y_pred) / len(y_pred)
                            recalls.append(float(recall))

                        # Save with standard thresholds
                        with open(threshold_data_path, "w") as f:
                            json.dump({
                                "thresholds": standard_thresholds.tolist(),
                                "recalls": recalls,
                                "dataset": test_d,
                                "num_samples": len(y_true),
                                "seed": cfg.seed,
                            }, f)

                        print("threshold arguments")
                        mlflow.log_artifact(threshold_data_path, "threshold_data")

            # Process dataset groups if provided
            merged_metrics = {}
            merged_raw_scores = {}

            if dataset_groups:
                for group_name, datasets in dataset_groups.items():
                    # Skip groups that include the main dataset
                    if main_dataset in datasets:
                        log.warning(f"Skipping group {group_name} as it contains the main dataset {main_dataset}")
                        continue

                    # Check if all datasets in the group exist
                    if not all(ds in all_metrics for ds in datasets):
                        missing = [ds for ds in datasets if ds not in all_metrics]
                        log.warning(f"Skipping group {group_name} as datasets {missing} dont exist")
                        continue

                    # Get dataset sizes for proper weighting
                    dataset_sizes = [dataset_sample_sizes[ds] for ds in datasets]

                    # Merge metrics by weighting to dataset sizes
                    merged_metrics[group_name] = merge_metrics(
                        [all_metrics[ds] for ds in datasets],
                        dataset_sizes,
                    )

                    # Merge raw scores by concatenation
                    merged_y_true = np.concatenate([all_raw_scores[ds]["y_true"] for ds in datasets])
                    merged_y_score = np.concatenate([all_raw_scores[ds]["y_score"] for ds in datasets])
                    merged_raw_scores[group_name] = {
                        "y_true": merged_y_true,
                        "y_score": merged_y_score,
                    }

                    # Log merged metrics
                    for k, v in merged_metrics[group_name].items():
                        mlflow.log_metric(f"{group_name}_{k}", float(v))

                    # For merged datasets, also calculate and log threshold data
                    if main_dataset and main_dataset in all_roc_data:
                        main_thresholds = all_roc_data[main_dataset]["thresholds"]

                        # Calculate recalls for all thresholds
                        merged_recalls = []
                        for threshold in main_thresholds:
                            y_pred = merged_y_score >= threshold

                            # Calculate recall (all samples are fraud in these datasets)
                            recall = np.sum(y_pred) / len(y_pred)
                            merged_recalls.append(float(recall))

                        # Create unique file path in the temp directory
                        merged_data_path = os.path.join(temp_dir, f"{group_name}_threshold_data.json")

                        # Save threshold data as a JSON artifact with sample size information
                        with open(merged_data_path, "w") as f:
                            json.dump({
                                "thresholds": main_thresholds.tolist(),
                                "recalls": merged_recalls,
                                "dataset": group_name,
                                "constituent_datasets": datasets,
                                "dataset_sizes": dataset_sizes,
                                "total_samples": sum(dataset_sizes),
                                "run_id": run_id,
                                "seed": cfg.seed,
                            }, f)
                        mlflow.log_artifact(merged_data_path, "threshold_data")

    finally:
        # Clean tmp dir
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                log.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            log.warning(f"Error cleaning up temporary directory: {e}")


dataset_groups = {
    "static template swaping": [
        "swap",
        "swap_three",
    ],
    "dynamic template": [
        "double_sticker",
        "holo-completemask",
        "holo-star-world",
        "leaf-holo",
        "plain-holo",
    ],
    "static template": [
        "laser",
        "no-holo",
        "plastified-led",
        "plastified-lowreflect",
        "plastified-noholo",
    ],
    "MIDV-DynAttack": [
        "swap",
        "swap_three",
        "double_sticker",
        "holo-completemask",
        "holo-star-world",
        "leaf-holo",
        "plain-holo",
        "laser",
        "no-holo",
        "plastified-led",
        "plastified-lowreflect",
        "plastified-noholo",
    ],
}


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    """Entry point for testing."""
    seed_everything(cfg.seed, workers=True)
    task_name_full = HydraConfig.get().runtime.choices.decision + "_" + cfg.task_name
    if cfg.get("training", ""):
        task_name_full += cfg.training.trainer.run_name
    cfg.tuner.experiment_name = task_name_full
    tuner = instantiate(cfg.tuner)

    run = tuner.get_best_run()

    params = mlruntodict(run.data.params)
    params = DictConfig(params)

    mlflow.set_experiment("test_" + cfg.paths.split_name)
    if not already_run(cfg, "test_" + cfg.paths.split_name):
        pipeline_with_roc(cfg, params, task_name_full, log, dataset_groups)
    else:
        log.info("this run was already lauched")


if __name__ == "__main__":
    main()
