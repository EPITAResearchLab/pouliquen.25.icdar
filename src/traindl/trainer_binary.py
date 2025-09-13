import random
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from PIL import Image
from os.path import join as pjoin 
import os
import torch.nn.functional as F
from pathlib import Path


class BinaryDatasetSplit(torch.utils.data.Dataset):
    IMAGES_TRANSFORM = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
    def __init__(self,
                 base_path,
                 split_path,
                 split_name,
                 transform=None,
                 skip: int = 1):
        print(base_path, split_path, split_name)
        files_origins, files_frauds, frauds_type = self.get_origins_frauds(base_path, split_path, split_name, skip=skip)
        self.imgs = files_origins + files_frauds
        self.labels = [1 for _ in range(len(files_origins))] + [0 for _ in range(len(files_frauds))]
        self.labels_precise = ["origins" for _ in range(len(files_origins))] + frauds_type

        self.transform = transform
        print(self.transform)
        print("skipping", skip)

    def get_origins_frauds(self, base_path, split_path, split_name, skip=1):
        videos_names = (Path(split_path) / split_name).open().read().splitlines()
        type_doc = "origins"
        input_dir = Path(base_path) / type_doc
        files = []
        for v in videos_names:

            vid_path = input_dir / v
            if vid_path.suffix:
                vid_path = vid_path.parent
            if not vid_path.exists():
                continue
            imgs = [im_p for i, im_p in enumerate(sorted(vid_path.glob("*.jpg")))
                    if not i % skip]

            files.extend(imgs)

        files_fraud = []
        fraud_types = []
        for fraud in ["fraud/photo_holo_copy", "fraud/pseudo_holo_copy", "fraud/copy_without_holo"]:
            input_dir = Path(base_path) / fraud
            for v in videos_names:

                vid_path = input_dir / v
                if vid_path.suffix:
                    vid_path = vid_path.parent
                if not vid_path.exists():
                    continue
                imgs = [im_p for i, im_p in enumerate(sorted(vid_path.glob("*.jpg")))
                        if not i % skip]

                files_fraud.extend(imgs)
                fraud_types += [fraud for _ in range(len(imgs))]
        print(len(files), len(files_fraud))
        return files, files_fraud, fraud_types

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

    def __len__(self):
        return len(self.imgs)

    def randomFlipRotation(self, img):
        op = random.choice(self.IMAGES_TRANSFORM)
        return img.transpose(op)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        label = self.labels[idx]

        if random.random() < 0.5:
            image = self.randomFlipRotation(image)

        if self.transform:
            image = self.transform(image)
        return image, label


class MIDVHoloBinaryDataModule(pl.LightningDataModule):
    def __init__(self,
                 input_dir: str,
                 split_dir: str,
                 transform = None,
                 batch_size: int = 32,
                 num_workers: int = 16,
                 skip: int = 1):
        super().__init__()
        self.data_dir = input_dir
        self.batch_size = batch_size
        print(transform)
        self.transform = transform
        self.split_dir = split_dir
        self.num_workers = num_workers
        self.skip = skip

    def setup(self, stage: str):
        self.midvholo_train = BinaryDatasetSplit(self.data_dir, self.split_dir, "train.txt", self.transform, self.skip)
        self.midvholo_val = BinaryDatasetSplit(self.data_dir, self.split_dir, "val.txt", self.transform, self.skip)

    def train_dataloader(self):
        return DataLoader(self.midvholo_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.midvholo_val, batch_size=self.batch_size, num_workers=self.num_workers)


class BinaryClassifier(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.backbone = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # train loop
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # test loop
        x, y = batch
        y_hat = self.backbone(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # val loop
        x, y = batch
        y_hat = self.backbone(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.lr)
        return optim

class Trainer:
    def __init__(self, epochs_max, model, seed=0, lr=0.001, accelerator="cuda", checkpoint_callback=None, run_name="") -> None:
        torch.set_float32_matmul_precision('high')
        pl.seed_everything(seed, workers=True)
        self.seed = seed
        self.model = model
        self.epochs = epochs_max
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback
        self.run_name = run_name
        self.decision = None
        self.val_fullvid = None
        self.lr = lr

    def val(self, general_model, decision, val_fullvid):
        self.decision = decision
        self.val_fullvid = val_fullvid
        self.general_model = general_model

    def train(self, datamodule, task_name):
        model = BinaryClassifier(self.model, self.lr)
        tags = {"task_name": task_name}
        checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_loss", filename='{epoch}-{val_loss:.2f}')
        mllogger = MLFlowLogger(log_model=True, run_name=f"{task_name}_{self.run_name}", tags=tags)
        run_id = mllogger.run_id
        mllogger.experiment.log_param(run_id, "lr", model.lr)
        mllogger.experiment.log_param(run_id, "transform", datamodule.transform)
        mllogger.experiment.log_param(run_id, "batch_size", datamodule.batch_size)
        loggers = [mllogger]
        # callbacks=([self.checkpoint_callback] if self.checkpoint_callback is not None else None)
        pl.seed_everything(self.seed, workers=True)
        trainer = pl.Trainer(max_epochs=self.epochs, accelerator=self.accelerator, logger=loggers, callbacks=[checkpoint], deterministic=True)
        trainer.fit(model, datamodule)

        # saving best model backbone for latter use
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(best_model_path)
        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        torch.save(model.backbone.state_dict(), os.path.join(os.path.dirname(best_model_path), f"backbone_{os.path.basename(best_model_path)}"))
        mllogger.experiment.log_metric(run_id, "best_val_loss", checkpoint.best_model_score)
        mllogger.experiment.log_param(run_id, "best_model_name", f"backbone_{os.path.basename(best_model_path)}")
        mllogger.experiment.log_param(run_id, "epochs_max", self.epochs)