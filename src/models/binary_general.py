from __future__ import annotations

import numpy as np
import torch


class ClassificationModelGeneral:
    """Classification Model."""
    i = -1
    MIN_SIZE = 2

    def __init__(self, model: any, accelerator: str = "cuda") -> None:
        """Initialize."""
        self.model = model
        self.model.eval()

        self.device = torch.device(accelerator)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.reset()

    def reset(self) -> None:
        """Reset var."""
        self.i = -1
        self.preds = np.array([])

    def apply(self, img_t: torch.tensor) -> float:
        """Apply a new tensor and returns the class.

        Returns:
            Classification class
        """
        self.i += 1

        with torch.no_grad():
            pred = (
                self.model(img_t.unsqueeze(0).to(self.device))
                .cpu()
                .softmax(dim=1)
                .argmax(dim=1)
                .item()
            )
            self.preds = np.append(self.preds, pred)
        if self.preds.size > self.MIN_SIZE:
            return self.preds.mean()
        return 0
