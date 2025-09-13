import json
from os.path import basename, dirname, normpath
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import torchvision.transforms as T
from PIL import Image


class MIDVHolov2PseudoLabel:
    MIN_FRAMES = 10

    def __init__(
        self,
        input_dir,
        split_file,
        skip: int = 1,
    ) -> None:

        self.skip = skip
        self.input_dir = Path(input_dir)
        vid_names = Path(split_file).open().read().splitlines()
        self.vids_res = []
        self.labels_vid = []

        for p in ["origins",
                  "fraud/copy_without_holo",
                    "fraud/photo_holo_copy",
                    "fraud/pseudo_holo_copy"]:
            for vid_name in vid_names:
                v_p = self.input_dir / p / Path(vid_name) / "video_infos.json"
                if v_p.exists():
                    self.labels_vid.append(p)
                    infos = json.load(v_p.open())
                    self.vids_res.append([i in infos["valid_frames"] for i in range(len(infos["video_path"])) if not i % skip])
        self.vid_names = vid_names
        print(self.skip)
        print()

    def getFraudNames(self):
        return [k for k in self.labels_dict.keys() if k.startswith("fraud")]

    def getVideos(self, idx):
        return self.videos[idx]

    def __getitem__(self, idx):
        return self.vids_res[idx]

    def isFraud(self, idx):
        return "fraud" in self.labels_vid[idx]

    def __len__(self) -> int:
        return len(self.vids_res)


class GeneralFakeDatasetPseudoLabel:
    def __init__(
        self,
        input_dir,
        split_file,
        skip: int = 1,
    ) -> None:

        self.skip = skip
        self.input_dir = Path(input_dir)
        vid_names = Path(split_file).open().read().splitlines()
        self.vids_res = []
        self.labels_vid = []

        for vid_name in vid_names:
            v_p = self.input_dir / Path(vid_name) / "video_infos.json"
            if v_p.exists():
                infos = json.load(v_p.open())
                self.vids_res.append([i in infos["valid_frames"] for i in range(len(infos["video_path"])) if not i % skip])
        self.vid_names = vid_names
        print(self.skip)
        print()

    def getFraudNames(self):
        return [k for k in self.labels_dict.keys() if k.startswith("fraud")]

    def getVideos(self, idx):
        return self.videos[idx]

    def __getitem__(self, idx):
        return self.vids_res[idx]

    def isFraud(self, idx):
        return True

    def __len__(self) -> int:
        return len(self.vids_res)
