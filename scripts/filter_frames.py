import json
import cv2
from pathlib import Path
import numpy as np
import tqdm
import sys


def main(bottom_th=5, safety_zone=1):
    main_path = Path(sys.argv[1])

    for vid_path in tqdm.tqdm(list(main_path.glob("*/*/*/") if "fraud" in main_path.name else main_path.glob("*/*/"))):
        images_path = sorted(vid_path.glob("*.jpg"))
        vid_infos = {
            "valid_frames": [],
            "too_bright_frames": [],
            "too_black_frames": [],
            "safety_zone": [],
            "frames": [],
            "params": {
                "bottom_threshold": bottom_th,
                "safety_zone": safety_zone,
            },
        }
        if len(images_path) < 20:
            continue
        vid_infos["frames"] = [im_p.name for im_p in images_path]

        images = [cv2.imread(str(im_p)) for im_p in images_path]
        threshold = images[0][..., 0].size / 3.0

        for i, im in enumerate(images):
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            mask = gray > bottom_th

            if mask.sum() > threshold:
                vid_infos["too_bright_frames"].append(i)
            elif (gray <= bottom_th).sum() > 0.99999 * gray.size:
                # equivalent to 1 pixel not black
                vid_infos["too_black_frames"].append(i)
            else:
                vid_infos["valid_frames"].append(i)
            
        security = set()
        for j in vid_infos["too_bright_frames"]:
            security.update(range(j - safety_zone + 1, j + safety_zone + 1))

        vid_infos["valid_frames"] = [
            i for i in vid_infos["valid_frames"] if i not in security
        ]

        vid_infos["safety_zone"] = list(security)

        with (vid_path / "video_infos.json").open("w") as f:
            json.dump(vid_infos, f)

        np.save(vid_path / "valid_frames.npy", vid_infos["valid_frames"])


if __name__ == "__main__":
    main()
