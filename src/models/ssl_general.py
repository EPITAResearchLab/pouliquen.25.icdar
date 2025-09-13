from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


class SSLGeneral:
    """Weakly supervised general model."""

    i = -1
    mask_holo_coarse = None
    embeddings = np.array([])
    diffs = np.array([])
    IMAGENET_NORMALIZE: ClassVar[dict[str : list[float]]] = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
    MIN_EMB = 2

    def __init__(self, model: any, accelerator: str = "cuda", input_size: int = 224, method: int = 0, metric: str = "cosine") -> None:
        self.model = model
        self.model.eval()

        accelerator = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model

        self.device = torch.device(accelerator)
        print(accelerator, self.device)
        self.model.to(self.device)
        self.metric = metric  # or cosine
        self.method = self.mean_sim

        self.transform = T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=self.IMAGENET_NORMALIZE["mean"],
                    std=self.IMAGENET_NORMALIZE["std"],
                ),
            ],
        )

    def nscore(self):
        n = 3
        if self.embeddings.shape[0] < n:
            return None
        base_emb = self.embeddings[-1].reshape(1, -1)
        res = []
        for i in range(-2, -n - 1, -1):
            embedding2 = self.embeddings[i].reshape(1, -1)
            res += [cosine_similarity(base_emb, embedding2)[0, 0]] * -i
        return np.mean(res)

    def mean_sim(self):
        if self.embeddings.shape[0] <= 2:
            return None

        similarity = pairwise_distances(self.embeddings, metric=self.metric)
        np.fill_diagonal(similarity, 0)
        return similarity.mean()

    def reset(self) -> None:
        self.i = -1
        self.h_percent = 0
        self.embeddings = np.array([])
        self.diffs = np.array([])

    def apply(self, img_t: torch.tensor) -> float | None:
        """"""
        self.i += 1

        with torch.no_grad():
            embedding = (
                self.model(img_t.unsqueeze(0).to(self.device))
                .flatten(start_dim=1)
                .cpu()
                .numpy()
            )

            if self.embeddings.size == 0:
                self.embeddings = embedding
            else:
                self.embeddings = np.concatenate((self.embeddings, embedding))
        if self.method == self.mean_sim:
            return self.method() if self.embeddings.shape[0] > self.MIN_EMB else None

        if self.embeddings.shape[0] > 1:
            diff = self.method()
            if diff is not None:
                self.diffs = np.append(self.diffs, diff)

        if self.diffs.size > 1:
            return 1 - np.median(self.diffs)
        return None

    def get_vid_embeddings(self, ims_t: torch.tensor, batch_n: int = 8) -> np.array:
        imgs_b = []
        embeddings = None
        with torch.no_grad():
            for im_t in ims_t:
                imgs_b.append(im_t)
                if len(imgs_b) == batch_n:
                    res = (
                        self.model(torch.stack(imgs_b).cuda())
                        .flatten(start_dim=1)
                        .cpu()
                    )
                    if embeddings is None:
                        embeddings = res
                    else:
                        embeddings = torch.cat((embeddings, res))
                    imgs_b = []

            if len(imgs_b) != 0:
                res = self.model(torch.stack(imgs_b).cuda()).flatten(start_dim=1).cpu()
                embeddings = res if embeddings is None else torch.cat((embeddings, res))
        return embeddings.numpy()
