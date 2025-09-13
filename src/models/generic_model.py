import logging

import timm
import torch
from torch import nn

log = logging.getLogger(__name__)


class GenericBackbone(nn.Module):
    """Generic model using timm."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        model_path: str = "",
        num_classes: int = -1,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=(pretrained and len(model_path) == 0),
            num_classes=num_classes,
        )

        if model_path is not None and len(model_path) != 0:
            log.info("loading checkpoint from path %s", model_path)
            self.load_state_dict(torch.load(model_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the net.

        Args:
            x: A tensor.

        Returns:
            A tensor.

        """
        return self.backbone(x)
