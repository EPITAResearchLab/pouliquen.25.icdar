from __future__ import annotations

import numpy as np
import tqdm

from src.utils.utils import get_metrics_from_values


class Cumulative:
    """Cumulative class, stops when above a threshold."""
    def __init__(self, th: float, min_frames: int = 5) -> None:
        """Initializes."""
        self.th = th
        self.min_frames = min_frames

    def process_frame_by_frame(self, frames, methodClass) -> tuple[bool, int]:
        """Process frames one by one.

        Returns:
            above threshold and the frame number.
        """
        # frames
        for i, frame in enumerate(frames):
            t = methodClass.apply(frame)
            if i >= self.min_frames and t >= self.th:  # 1 is for no holo
                return False, i
        return True, i

    def process(self, frames, methodClass) -> float:
        # frames
        res = np.array([])
        for i, frame in enumerate(frames):
            t = methodClass.apply(frame)
            if i >= self.min_frames:
                res = np.append(res, t)
        return res

    def getResults(self, dataset, methodClass) -> tuple[np.array, np.array]:
        metrics_frauds = np.array([])
        metrics_origins = np.array([])
        for i in tqdm.tqdm(range(len(dataset))):
            methodClass.reset()
            res = self.process(dataset[i], methodClass)
            if dataset.isFraud(i):
                metrics_frauds = np.append(metrics_frauds, res.max())
            else:
                metrics_origins = np.append(metrics_origins, res.max())

        return metrics_frauds, metrics_origins

    def tune(self, data, methodClass) -> tuple[float, int]:
        values_frauds, values_origins = self.getResults(data, methodClass)

        metrics_m = {"fscore": -1}
        th = 0

        full = np.concatenate((values_frauds, values_origins, [0, 1]))
        for i in np.unique(full):
            origin = values_origins < i
            frauds = values_frauds < i
            metrics = get_metrics_from_values(
                origin, frauds,
            )  # tp is a frauds predicted as fraud
            if metrics["fscore"] > metrics_m["fscore"]:
                th = i
                metrics_m = metrics
        self.th = th
        return metrics_m, th


class NDecision:
    """Decision after n frames."""
    def __init__(self, th: float, n_frames: int = -1) -> None:
        """Initializes."""
        self.th = th
        self.n_frames = -1
        if n_frames > 0:
            print(f"taking first {n_frames} frames")
            self.n_frames = n_frames - 1  # start at 0

    def process(self, frames, methodClass) -> tuple[float, int]:
        for i, frame in enumerate(frames):
            t = methodClass.apply(frame)
            if self.n_frames != -1 and i >= self.n_frames:
                return t, i - 1  # to test
        return t, i

    def process_frame_by_frame(self, frames, methodClass) -> tuple[float, int]:
        """Process frame by frame.

        Returns:
            Threshold and frame number
        """
        t, i = self.process(frames, methodClass)
        return t < self.th, i  # 1 is for no holo

    def getResults(self, dataset, methodClass) -> tuple[np.array, np.array]:
        """Get results applying a method on a dataset."""
        metrics_frauds = np.array([])
        metrics_origins = np.array([])
        for i in tqdm.tqdm(range(len(dataset))):
            methodClass.reset()
            res, _ = self.process(dataset[i], methodClass)

            if dataset.isFraud(i):
                metrics_frauds = np.append(metrics_frauds, res)
            else:
                metrics_origins = np.append(metrics_origins, res)

        return metrics_frauds, metrics_origins

    def tune(self, data: any, method_class: any) -> tuple[dict, float]:
        """Tune a method on a dataset.

        Returns:
            a tuple with metrics and best threshold
        """
        values_frauds, values_origins = self.getResults(data, method_class)
        metrics_m = {"fscore": -1}
        th = 0

        full = np.concatenate((values_frauds, values_origins, [0, 1, 1.1]))
        for i in np.unique(full):
            # tp is a frauds predicted as fraud
            origin = values_origins < i
            frauds = values_frauds < i
            metrics = get_metrics_from_values(origin, frauds)
            if metrics["fscore"] > metrics_m["fscore"]:
                th = i
                metrics_m = metrics

        self.th = th
        return metrics_m, th
