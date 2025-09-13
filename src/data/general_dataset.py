from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class GeneralDataset:
    """General dataset that could be used by any dataset."""

    IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    MIN_FRAMES = 10

    def __init__(self, videos_file_path: str, transform: any, fraud: bool = False, skip: int = 3) -> None:
        self.videos = []
        videos_file_path = Path(videos_file_path)
        self.input_dir = videos_file_path.parent
        vids_paths = videos_file_path.read_text().splitlines()

        for v in vids_paths:
            imgs = (self.input_dir / v / "files.txt").read_text().splitlines()
            if len(imgs) > self.MIN_FRAMES:
                self.videos.append(imgs)
            else:
                print(f"not enought images found {v}")
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(
                mean=self.IMAGENET_NORMALIZE["mean"],
                std=self.IMAGENET_NORMALIZE["std"],
            ),
        ])
        self.transform = T.Compose([
            T.Resize(230),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.1, 0.1, 0.1],  # Dark image specific normalization
                        std=[0.2, 0.2, 0.2]),
        ])
        self.transform = transform
        self.fraud = fraud
        print(f"{self.transform} {self.fraud=}")
        self.skip = skip

    def __getitem__(self, idx: int) -> torch.tensor:
        """Get items.

        Yields:
            Image tensor
        """
        for i, im_p in enumerate(self.videos[idx]):
            if i % self.skip:  # only use (total video images / skip) images
                continue
            im = Image.open(self.input_dir / im_p).convert("RGB")
            im_t = self.transform(im) if self.transform is not None else im
            yield im_t

    def isFraud(self, idx: int):
        return self.fraud

    def __len__(self) -> int:
        """Number of videos."""
        return len(self.videos)
