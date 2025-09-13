from __future__ import annotations

import cv2
import numpy as np


class MIDVBaseline:
    """Reproduction of MIDV Holo paper.

    https://www.researchgate.net/publication/373232277_MIDV-Holo_A_Dataset_for_ID_Document_Hologram_Detection_in_a_Video_Stream
    """
    C = None
    h = 0
    i = -1

    def __init__(
        self, hight_threshold: float, s_t: float, t: int, n_min: int = 4,
    ) -> None:
        """Initialize the MIDV Holo algo."""
        self.th_white = hight_threshold
        self.n_min = n_min
        self.reset()

        self.s_t = s_t
        self.t = t

    def reset(self) -> None:
        """Reset all var."""
        self.C = None
        self.h = 0
        self.i = -1
        self.h_percent = 0

    def normalize_gray_world(self, img: np.array) -> tuple[np.array, np.array]:
        """Gray world normalization.

        Returns:
            The normalized image and the mask.
        """
        img_array = img.copy()
        img_array = img_array.astype(np.float32)

        # mask of highlight
        mask = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) <= self.th_white

        b_mean, g_mean, r_mean = (
            np.mean(img_array[:, :, 0][mask]),
            np.mean(img_array[:, :, 1][mask]),
            np.mean(img_array[:, :, 2][mask]),
        )

        global_mean = np.mean([r_mean, g_mean, b_mean])

        img_array[:, :, 0] = img_array[:, :, 0] * global_mean / b_mean
        img_array[:, :, 1] = img_array[:, :, 1] * global_mean / g_mean
        img_array[:, :, 2] = img_array[:, :, 2] * global_mean / r_mean

        img_array = np.clip(img_array, 0, 255)

        img_array[~mask] = 0

        return img_array.astype(np.uint8), mask

    def apply(self, img: np.array) -> float:
        """Apply The MIDV Holo algo for one image.

        Returns:
            Current percentage of hologram.
        """
        self.i += 1
        if self.C is None:
            self.C = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float64)
            self.smax = np.zeros((img.shape[0], img.shape[1]))
            self.n = np.zeros((img.shape[0], img.shape[1]))

        img_n, mask = self.normalize_gray_world(img)

        h, s, _ = cv2.split(cv2.cvtColor(img_n, cv2.COLOR_BGR2HSV))
        h = h.astype(np.float64)
        s = img_n.max(axis=2) - img_n.min(axis=2)

        self.smax = np.max(np.stack([self.smax, s]), axis=0)
        self.smax[~mask] = 0

        self.C[:, :, 0][mask] += s[mask] * np.cos(np.deg2rad(h[mask] * 2))
        self.C[:, :, 1][mask] += s[mask] * np.sin(np.deg2rad(h[mask] * 2))
        self.n += mask

        c_f = self.C.copy()

        n_min = self.i // 2
        c_f[:, :, 0][self.n > n_min] /= self.n[self.n > n_min]
        c_f[:, :, 1][self.n > n_min] /= self.n[self.n > n_min]

        c_map = np.linalg.norm(c_f, axis=-1)
        smax_tmp = self.smax.copy()
        smax_tmp[self.n <= n_min] = 0

        s_th = smax_tmp > self.s_t
        i_no = np.ones(s_th.shape) * 255
        i_no[s_th] = (c_map[s_th] / smax_tmp[s_th]) * 255
        i_no_bin = i_no > self.t

        self.h_percent = i_no_bin.sum() / i_no_bin.size

        return 1 - self.h_percent
