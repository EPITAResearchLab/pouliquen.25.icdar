from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class GeneralDatasetMIDVv2:
    """General dataset that could be used by any dataset."""

    MIN_FRAMES = 10

    def __init__(self, input_dir: str,
                 split_file: str,
                 transform: any,
                 fraud: bool = True,
                 skip: int = 1) -> None:
        self.videos = []
        self.input_dir = Path(input_dir)
        videos_names = Path(split_file).open().read().splitlines()
        for v in videos_names:

            vid_path = self.input_dir / v
            if vid_path.suffix:
                vid_path = vid_path.parent
            if not vid_path.exists():
                continue
            imgs = sorted(vid_path.glob("*.jpg"))
            if len(imgs) > self.MIN_FRAMES:
                self.videos.append(imgs)
        self.transform = T.Compose([
            T.Resize(230),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.1, 0.1, 0.1],  # Dark image specific normalization
                        std=[0.2, 0.2, 0.2]),
        ])
        self.transform = transform
        self.skip = skip
        print(self.transform)
        self.fraud = fraud
        print(f"{skip=} ${input_dir}")
        print(f"label fraud {self.fraud}")

    def __getitem__(self, idx: int) -> torch.tensor:
        """Get items.

        Yields:
            Image tensor
        """
        for i, im_p in enumerate(self.videos[idx]):
            if i % self.skip:  # only use (total video images / skip) images
                continue
            im = Image.open(str(im_p)).convert("RGB")
            im_t = self.transform(im) if self.transform is not None else im
            yield im_t

    def isFraud(self, idx: int):
        return self.fraud

    def __len__(self) -> int:
        """Number of videos."""
        return len(self.videos)
