from __future__ import annotations

import os
import random
from os.path import join as pjoin
from pathlib import Path

import lightning as pl
import PIL
import PIL.Image
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


class MIDVHoloDataset:
    IMAGES_TRANSFORM: tuple[int, ...] = [
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270,
    ]

    def __init__(
        self,
        input_dir: str,
        transform,
        split_dir: str = "",
        split_file: str = "train.txt",
        only_label: bool | None = None,
        flip_rot: bool = True,
        skip: int = 1,
    ) -> None:
        """Initialise the dataset for training."""
        self.transform = transform
        self.labels_dict = {
            "fraud/copy_without_holo": {},
            "fraud/photo_holo_copy": {},
            "fraud/pseudo_holo_copy": {},
            "origins": {},
        }
        self.shorttopath = {
            "copy_without_holo": "fraud/copy_without_holo",
            "photo_holo_copy": "fraud/photo_holo_copy",
            "pseudo_holo_copy": "fraud/pseudo_holo_copy",
            "origins": "origins",
        }
        self.fraud_names = [k for k in self.labels_dict if k != "origins"]
        self.files = []
        self.labels = []
        self.input_dir = os.path.normpath(input_dir)
        self.only_label = only_label
        for la in self.labels_dict:
            files_tmp, labels_tmp = self.get_files_split(
                Path(self.input_dir) / la, split_dir, split_file, skip=skip,
            )
            self.files += files_tmp
            self.labels += labels_tmp
        self.lenght = self.__len__()
        self.flip_rot = flip_rot
        self.skip = skip
        if self.flip_rot:
            print("random flip and rotation")
        print(self.lenght, "skipping", self.skip)

    def random_flip_rotation(self, imgs: list[PIL.Images]) -> list[PIL.Images]:
        """Randomly apply a flip or rotation.

        Returns
        -------
        """
        op = random.choice(self.IMAGES_TRANSFORM)
        return [img.transpose(op) for img in imgs]

    def get_files_split(
        self,
        input_dir: Path,
        split_dir: str,
        split_file: str = "",
        skip: int = 1,
    ) -> tuple[list[str], list[str]]:
        """Get the files corresponding to the split.

        Returns:
            images and corresponding labels
        """
        images = []
        labels = []
        general_type = input_dir.name
        with open(pjoin(split_dir, split_file)) as f:
            video_names = f.read().split("\n")
        for vn in video_names:
            vn = input_dir / vn
            name = (
                general_type if general_type == "origins" else "fraud/" + general_type
            )
            if (
                self.only_label is not None
                and (general_type == "origins") != self.only_label
            ):
                continue
            l = f"{name}/{vn.name}"

            tmp_lst = [im_p for i, im_p in enumerate(sorted(vn.glob("*.jpg")))
                       if not i % skip]
            images += tmp_lst
            labels += [l] * len(tmp_lst)
            self.labels_dict[name][l] = tmp_lst
        if len(images) != len(labels):
            msg = "Images must be the same size as labels."
            raise ValueError(msg)
        return images, labels

    def __getitem__(
        self, idx: int,
    ) -> tuple[list[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], str]:
        """Return tensors for anchor, positive and negative.

        Returns:
            tuple of anchor, positive and negative and the label of the anchor.

        """
        f = self.files[idx]
        l = self.labels[idx]
        if "origins" in l:
            im = Image.open(pjoin(self.input_dir, l, f))
            tmp_l = self.labels[idx + 1 if idx + 1 < self.lenght else idx - 1]
            if tmp_l == l:
                im_n = Image.open(
                    pjoin(
                        self.input_dir,
                        tmp_l,
                        self.files[idx + 1 if idx + 1 < self.lenght else idx - 1],
                    ))
            else:
                im_n = Image.open(
                    pjoin(self.input_dir, self.labels[idx - 1], self.files[idx - 1])
                )

            if self.flip_rot and random.random() < 0.5:
                im, im_n = self.random_flip_rotation((im, im_n))

            return [self.transform(im), self.transform(im), self.transform(im_n)], l

        im = Image.open(pjoin(self.input_dir, l, f))
        fraud = "/".join(l.split("/")[:2])
        img_path_tmp = random.choice(self.labels_dict[fraud][l])
        im_p = Image.open(pjoin(self.input_dir, l, img_path_tmp))
        possible_frauds = [k for k in self.fraud_names if k != fraud]

        fraud_n = random.choice(possible_frauds)
        k_n = fraud_n + "/" + "/".join(l.split("/")[2:])
        im_n = random.choice(self.labels_dict[fraud_n][k_n])
        im_n = Image.open(pjoin(self.input_dir, k_n, im_n))

        if self.flip_rot and random.random() < 0.5:
            im, im_p, im_n = self.random_flip_rotation((im, im_p, im_n))

        return [self.transform(im), self.transform(im_p), self.transform(im_n)], l

    def __len__(self) -> int:
        """Lenght.

        Returns:
            lenght of the images.

        """
        return len(self.files)


class MIDVHoloDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_dir: str,
        split_dir: str,
        transform: bool | None = None,
        batch_size: int = 32,
        num_workers: int = 15,
        only_label: bool | None = None,
        flip_rot: bool = True,
        skip: int = 1,
    ):
        super().__init__()
        self.data_dir = input_dir
        self.batch_size = batch_size
        print(transform)
        self.transform = transform
        self.split_dir = split_dir
        self.num_workers = num_workers
        print("only label", only_label)
        self.only_label = only_label
        self.flip_rot = flip_rot
        self.skip = skip

    def setup(self, stage: str):
        self.midvholo_train = MIDVHoloDataset(
            self.data_dir,
            self.transform,
            self.split_dir,
            "train.txt",
            self.only_label,
            self.flip_rot,
            self.skip,
        )
        self.midvholo_val = MIDVHoloDataset(
            self.data_dir,
            self.transform,
            self.split_dir,
            "val.txt",
            self.only_label,
            self.flip_rot,
            self.skip,
        )

    def train_dataloader(self):
        return DataLoader(
            self.midvholo_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.midvholo_val, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ModelTriplet(pl.LightningModule):
    """Triplet Model."""

    def __init__(
        self, model, lr=0.1, general_model=None, decision=None, val_fullvid=None,
    ):
        """Initialize the Triplet Model."""
        super().__init__()
        self.backbone = model
        self.lr = lr
        self.criterion = nn.TripletMarginLoss()


        self.decision = decision
        self.val_fullvid = val_fullvid
        self.general_model = general_model

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass throught the backbone.

        Returns:
            tensor of logits.
        """
        return self.backbone(x).flatten(start_dim=1)

    def training_step(
        self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int
    ) -> torch.tensor:
        ancor_img, positive_img, negative_img = batch[0]

        anchor_out = self.forward(ancor_img)
        positive_out = self.forward(positive_img)
        negative_out = self.forward(negative_img)

        loss = self.criterion(anchor_out, positive_out, negative_out)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int
    ) -> torch.tensor:
        """Single validation step on a batch of data from training dataset.

        Returns:
            tensor of loss.
        """
        ancor_img, positive_img, negative_img = batch[0]

        anchor_out = self.forward(ancor_img)
        positive_out = self.forward(positive_img)
        negative_out = self.forward(negative_img)

        loss = self.criterion(anchor_out, positive_out, negative_out)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optim}


class Trainer:
    def __init__(
        self,
        epochs_max,
        model,
        seed=0,
        lr=0.1,
        accelerator="cuda",
        checkpoint_callback=None,
        run_name="",
    ) -> None:
        torch.set_float32_matmul_precision("medium")
        pl.seed_everything(seed, workers=True)
        self.seed = seed
        self.model = model
        self.epochs = epochs_max
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback
        self.run_name = run_name
        self.decision = None
        self.val_fullvid = None
        self.general_model = None
        self.lr = lr

    def val(self, general_model, decision, val_fullvid):
        self.decision = decision
        self.val_fullvid = val_fullvid
        self.general_model = general_model

    def train(self, datamodule, task_name):
        torch.cuda.empty_cache()
        model = ModelTriplet(
            self.model, self.lr,
            self.general_model, self.decision, self.val_fullvid,
        )
        tags = {"task_name": task_name}
        checkpoint = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            filename="{epoch}-{val_loss:.2f}",
        )
        mllogger = MLFlowLogger(
            log_model=True, run_name=f"{task_name}_{self.run_name}", tags=tags
        )
        run_id = mllogger.run_id
        mllogger.experiment.log_param(run_id, "lr", model.lr)
        mllogger.experiment.log_param(run_id, "transform", datamodule.transform)
        mllogger.experiment.log_param(run_id, "batch_size", datamodule.batch_size)
        loggers = [mllogger]
        # callbacks=([self.checkpoint_callback] if self.checkpoint_callback is not None else None)
        pl.seed_everything(self.seed, workers=True)
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            logger=loggers,
            callbacks=[checkpoint],
            deterministic=True,
        )  # devices=[1],

        trainer.fit(model, datamodule)

        # saving best model for latter use
        best_model_path = trainer.checkpoint_callback.best_model_path
        # print(best_model_path)
        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        #model.backbone.reset_classifier()  # remove classification layer
        torch.save(
            model.backbone.state_dict(),
            os.path.join(
                os.path.dirname(best_model_path),
                f"backbone_{os.path.basename(best_model_path)}",
            ),
        )
        mllogger.experiment.log_metric(
            run_id, "best_val_loss", checkpoint.best_model_score
        )
        mllogger.experiment.log_param(
            run_id, "best_model_name", f"backbone_{os.path.basename(best_model_path)}"
        )
        mllogger.experiment.log_param(run_id, "epochs_max", self.epochs)
