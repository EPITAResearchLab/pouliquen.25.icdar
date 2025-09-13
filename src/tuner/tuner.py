from __future__ import annotations

import mlflow


class SelectFromRun:
    """Select the best run."""

    def __init__(
        self, experiment_name: str, metrics_name: str = "metrics.fscore",
    ) -> None:
        """Initialize by selecting the best run from the experiments."""
        self.experiment_name = experiment_name
        client = mlflow.MlflowClient()
        current_experiment = dict(client.get_experiment_by_name(experiment_name))
        self.experiment_id = current_experiment["experiment_id"]
        self.runs = client.search_runs(
            [self.experiment_id], order_by=[f"{metrics_name} DESC"]
        )

    def get_best_run(
        self, task_name: str | None = None, params: str | None = None,
    ) -> dict | None:
        """Get the best run, by ordering and the selecting the best.

        Returns:
            run or None if no run is founded

        """
        if params is None:
            return self.runs[0]

        if task_name is not None:
            for run in self.runs:
                if run.data.task_name == task_name:
                    return run

        for run in self.runs:
            if run.data.params == {k: str(v) for k, v in params.items()}:
                return run
        return None
