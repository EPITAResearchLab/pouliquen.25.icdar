from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms as T

log = logging.getLogger(__name__)


class DummyClassification:
    """Dummy classification Model."""

    def __init__(self) -> None:
        """Initialize."""
        self.reset()

    def reset(self) -> None:
        """Reset all vars."""
        self.preds = np.array([])

    def apply(self, label: bool) -> float:
        """Apply a new tensor and returns the class.

        Returns:
            Classification class
        """

        self.preds = np.append(self.preds, int(label))
        # if self.preds.size > self.MIN_SIZE:
        return self.preds.mean()
