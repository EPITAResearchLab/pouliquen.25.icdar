from __future__ import annotations

import os
import random
from os.path import join as pjoin
from pathlib import Path

import joblib
import lightning as pl
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import griddata
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
from torch import optim
import cv2
import json

from tqdm import tqdm


class ElasticTransform:
    def __init__(self, alpha=50, sigma=5):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        img_np = np.array(img)
        shape = img_np.shape

        # Generate random displacement fields
        dx = np.random.rand(shape[0], shape[1]) * 2 - 1
        dy = np.random.rand(shape[0], shape[1]) * 2 - 1

        # Gaussian filter the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma) * self.alpha

        # Generate mesh grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Displacement fields
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # Apply displacement
        warped = cv2.remap(img_np, map_x, map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(warped)

class NonLinearDeformation:
    def __init__(self, grid_size=4, magnitude=0.2):
        self.grid_size = grid_size
        self.magnitude = magnitude

    def __call__(self, img):
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Create control point grid
        x_steps = np.linspace(0, w, self.grid_size)
        y_steps = np.linspace(0, h, self.grid_size)
        x_grid, y_grid = np.meshgrid(x_steps, y_steps)

        # Add random displacement to control points
        x_rand = x_grid + np.random.randn(*x_grid.shape) * w * self.magnitude
        y_rand = y_grid + np.random.randn(*y_grid.shape) * h * self.magnitude

        # Create full resolution grid
        x_full = np.linspace(0, w, w)
        y_full = np.linspace(0, h, h)
        x_map, y_map = np.meshgrid(x_full, y_full)

        # Interpolate displacement field
        grid_z = griddata((x_grid.flatten(), y_grid.flatten()), 
                         x_rand.flatten(), 
                         (x_map, y_map), 
                         method='cubic')
        grid_z[np.isnan(grid_z)] = x_map[np.isnan(grid_z)]
        map_x = grid_z

        grid_z = griddata((x_grid.flatten(), y_grid.flatten()), 
                         y_rand.flatten(), 
                         (x_map, y_map), 
                         method='cubic')
        grid_z[np.isnan(grid_z)] = y_map[np.isnan(grid_z)]
        map_y = grid_z

        # Apply displacement
        warped = cv2.remap(img_np, 
                          map_x.astype(np.float32),
                          map_y.astype(np.float32),
                          cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(warped)


class FrameDataset(Dataset):  # FrameDataset
    def __init__(self, frame_paths, labels, transform_valid=None, transform_fake=None):
        # super().__init__(frame_paths, labels, transform_valid, transform_fake)
        self.frame_paths = frame_paths
        self.labels = labels
        self.transform_valid = transform_valid
        self.transform_fake = transform_fake

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.frame_paths[idx]).convert('RGB')
        label = self.labels[idx]
        gray = np.array(image).mean(axis=-1)
        if (gray < 5).sum() > 0.9999 * gray.size or (gray > 5).sum() > 0.3 * gray.size:
            # print(f"{idx} changing to fake")
            label = 0

        if label == 1 and self.transform_valid:
            image = self.transform_valid(image)
        elif label == 0 and self.transform_fake:
            image = self.transform_fake(image)

        return image, label

class DescriptorDataset(FrameDataset):
    def __init__(self, frame_paths,
                 labels,
                 transform_valid=None,
                 transform_fake=None,
                 descriptor=None,
                 ):
        super().__init__(frame_paths, labels, transform_valid, transform_fake)
        self.descriptor = descriptor

    def __getitem__(self, idx: int):
        image_descr, label = super().__getitem__(idx)

        if self.descriptor is not None:
            image_descr = self.descriptor.extract_descriptor(np.array(image_descr))
        return image_descr, label


def get_frames(path, splitfile=None, fraud=False, skip_n=1):
    valid_frames = []
    other_frames = []
    video_path = Path(path)
    if splitfile:
        print(f"loading for {video_path.name} from", splitfile, f"skipping {skip_n} so {15/skip_n}fps")
        possible_dirs = [video_path / p for p in Path(splitfile).open().read().splitlines()]
    else:
        possible_dirs = list((video_path / "ID").iterdir()) + (
            list((video_path / "passport").iterdir())
            if (video_path / "passport").exists()
            else [])
    for video_path in possible_dirs:
        image_paths = sorted(video_path.glob("*.jpg"), key=str)
        if (len(image_paths) > 20 or (video_path / "valid_frames.npy").exists()):
            # valid_frames_i = np.load(video_path / "valid_frames.npy")
            video_infos = json.load((video_path / "video_infos.json").open())
            if fraud:
                other_frames += [im_p for i, im_p in enumerate(image_paths) if not i % skip_n]
            else:
                valid_frames += [im_p for i, im_p in enumerate(image_paths)
                                if i in video_infos["valid_frames"] and not i % skip_n]

                other_frames += [im_p for i, im_p in enumerate(image_paths)
                                if i in video_infos["too_bright_frames"] and not i % skip_n]
    return valid_frames, other_frames


def get_image_paths(data_path, split_path, split_name, legitonly=False, skip_n=1):
    path = f"{data_path}/origins/"
    split_file = f"{split_path}/{split_name}"
    legit_frames, fake_frames = get_frames(path, splitfile=split_file, skip_n=skip_n)

    if not legitonly:
        path = f"{data_path}/fraud/photo_holo_copy"
        tmp_valid, tmp_other = get_frames(path, splitfile=split_file, skip_n=skip_n)
        fake_frames += tmp_valid
        fake_frames += tmp_other

        path = f"{data_path}/fraud/copy_without_holo"
        tmp_valid, tmp_other = get_frames(path, splitfile=split_file, skip_n=skip_n)
        fake_frames += tmp_valid
        fake_frames += tmp_other

        path = f"{data_path}/fraud/pseudo_holo_copy"
        tmp_valid, tmp_other = get_frames(path, splitfile=split_file, skip_n=skip_n)
        fake_frames += tmp_valid
        fake_frames += tmp_other
    return legit_frames, fake_frames


class ValidNonValidDataModule:
    def __init__(self, input_dir: str,
                 split_dir,
                 transform_valid=None,
                 transform_nonvalid=None,
                 batch_size: int = 32,
                 num_workers=4,
                 legit_only=False,
                 notransformations=False):
        super().__init__()
        self.data_dir = input_dir
        self.batch_size = batch_size
        self.transform_valid = transforms.Compose([
            transforms.Resize(230),
            transforms.RandomCrop(224),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1, 0.1, 0.1],
                              std=[0.2, 0.2, 0.2]),
        ])

        self.transform_nonvalid = transforms.Compose([
            transforms.Resize(230),
            transforms.RandomCrop(224),

            transforms.RandomApply([
                # Base geometric transformations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(30),
                # Advanced deformations (before tensor conversion)
                transforms.RandomApply([ElasticTransform(alpha=60, sigma=6)], p=0.3),
                transforms.RandomApply([NonLinearDeformation(grid_size=4, magnitude=0.25)], p=0.3),

                # # Affine transformations
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=20,
                ),

                # Color and intensity transformations
            ], p=0.2),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.4,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1, 0.1, 0.1],
                              std=[0.2, 0.2, 0.2]),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1, 0.1, 0.1],  # Dark image specific normalization
                              std=[0.2, 0.2, 0.2]),
        ])

        self.split_dir = split_dir
        self.num_workers = num_workers
        self.legit_only = legit_only
        self.notransformations = notransformations

    def setup(self, stage: str):
        # Create datasets
        legit_frames, fake_frames = get_image_paths(self.data_dir, self.split_dir, "train.txt", skip_n=skip_n)

        legit_frames_val, fake_frames_val = get_image_paths(self.data_dir, self.split_dir, "val.txt", skip_n=skip_n)

        val_paths = legit_frames_val + fake_frames_val
        val_labels = np.array([1] * len(legit_frames_val) + [0] * len(fake_frames_val))

        train_paths = legit_frames + fake_frames
        train_labels = np.array([1] * len(legit_frames) + [0] * len(fake_frames))

        self.train_dataset = FrameDataset(
            train_paths,
            train_labels,
            transform_valid=self.transform_valid if self.transform_valid is not None else self.transform_val,
            transform_fake=self.transform_fake if self.transform_fake is not None else self.transform_val,
        )

        # Validation dataset uses only basic transforms for both classes
        val_dataset = FrameDataset(
            val_paths,
            val_labels,
            transform_valid=self.transform_val,
            transform_fake=self.transform_val,  # Use valid transforms for validation
        )

        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                   shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.midvholo_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.midvholo_val, batch_size=self.batch_size, num_workers=self.num_workers)


class FrameClassifier:
    def __init__(self, model, learning_rate=0.001, save_path="best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = self.model.to(self.device)
        self.save_path = save_path
        self.notransformations = False

        # Basic transformations for valid samples
        self.transform_valid = transforms.Compose([
            transforms.Resize(230),
            transforms.RandomCrop(224),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1, 0.1, 0.1],
                              std=[0.2, 0.2, 0.2]),
        ])

        # Aggressive transformations for fake samples
        self.transform_fake = transforms.Compose([
            transforms.Resize(230),
            transforms.RandomCrop(224),
            transforms.RandomApply([
                # geometric transformations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(30),
                # deformations
                transforms.RandomApply([ElasticTransform(alpha=60, sigma=6)], p=0.3),
                transforms.RandomApply([NonLinearDeformation(grid_size=4, magnitude=0.25)], p=0.3),

                # # Affine transformations
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=20,
                ),

            ], p=0.2),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.4,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1, 0.1, 0.1],
                              std=[0.2, 0.2, 0.2]),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1, 0.1, 0.1],
                              std=[0.2, 0.2, 0.2]),
        ])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def prepare_data(self, train_paths, train_labels, val_paths, val_labels, batch_size=32, num_workers=8, onlylegits=False, notransformations=False):
        """Prepare train and validation dataloaders"""
        print(len(train_paths), sum(train_labels))
        print(f"{notransformations=}")
        # Create datasets
        train_dataset = FrameDataset(
            train_paths,
            train_labels,
            transform_valid=self.transform_valid if not notransformations else self.transform_val,
            transform_fake=self.transform_fake if not notransformations else self.transform_val,
        )
        self.notransformations = notransformations


        # Validation dataset uses only basic transforms for both classes
        val_dataset = FrameDataset(
            val_paths,
            val_labels,
            transform_valid=self.transform_val,
            transform_fake=self.transform_val,  # Use valid transforms for validation
        )

        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

        return self.train_loader, self.val_loader

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(len(self.train_loader))
        for inputs, labels in tqdm(self.train_loader, total=len(self.train_loader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return epoch_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, total=len(self.val_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return val_loss, accuracy

    def train(self, num_epochs=10):
        best_val_acc = 0

        for epoch in range(num_epochs):
            # Regular training epoch
            print(f'Epoch {epoch+1}/{num_epochs}:')
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)

    def load(self) -> None:
        self.model.load_state_dict(torch.load(self.save_path))

    def predict(self, image_path):
        """Predict single image"""
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = self.transform_val(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()

        return pred, prob[0][pred].item()

    def predict_batch(self, image_paths, batch_size=32):
        """Predict multiple images in batches"""
        self.model.eval()

        # Initialize lists to store predictions and probabilities
        all_preds = []
        all_probs = []

        # Iterate over image paths in chunks of batch_size
        for i in range(0, len(image_paths), batch_size):
            # Get the current batch of image paths
            batch_image_paths = image_paths[i:i + batch_size]

            # Load and preprocess images
            images = []
            for image_path in batch_image_paths:
                image = Image.open(image_path).convert("RGB")
                image = self.transform_val(image)
                images.append(image)

            # Stack images into a single tensor
            images_batch = torch.stack(images).to(self.device)

            with torch.no_grad():
                output = self.model(images_batch)
                prob = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)

            # Convert predictions and probabilities to list
            preds = pred.tolist()
            probs = [prob[j][preds[j]].item() for j in range(len(preds))]

            # Append results to the overall lists
            all_preds.extend(preds)
            all_probs.extend(probs)

        return all_preds, all_probs

class DescriptorFrameClassifier:
    def __init__(self, descriptor, learning_rate=0.001, save_path="best_model.joblib"):
        self.save_path = save_path
        self.descriptor = descriptor

        # Initialize SGD classifier
        self.model = SGDClassifier(
            loss='log_loss',
            learning_rate='optimal',
            alpha=learning_rate,
            max_iter=1,
            warm_start=True,  # incremental learning
            random_state=42,
        )

        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # Keep the same transformations from original code
        self.transform_valid = transforms.Compose([
            transforms.Resize(230),
            transforms.RandomCrop(224),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),
        ])

        self.transform_fake = transforms.Compose([
            transforms.Resize(230),
            transforms.RandomCrop(224),
            transforms.RandomApply([
                # geometric transformations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(30),
                # deformations
                transforms.RandomApply([ElasticTransform(alpha=60, sigma=6)], p=0.3),
                transforms.RandomApply([NonLinearDeformation(grid_size=4, magnitude=0.25)], p=0.3),

                # Affine transformations
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0, 0.2),
                    scale=(0.8, 1.2),
                    shear=20,
                ),

            ], p=0.2),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.4,
            ),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
        ])


    def prepare_data(self, train_paths: list[str | Path],
                     train_labels: list[int],
                     val_paths: list[str | Path],
                     val_labels: list[int],
                     batch_size: int = 32,
                     num_workers: int = 8,
                     notransformations: bool = False,
                     ) -> tuple[DataLoader, DataLoader]:
        """Prepare train and validation dataloaders."""
        print(len(train_paths), len(train_labels))
        train_dataset = DescriptorDataset(
            train_paths,
            train_labels,
            transform_valid=self.transform_valid if not notransformations else self.transform_val,
            transform_fake=self.transform_fake if not notransformations else self.transform_val,
            descriptor=self.descriptor,
        )

        val_dataset = DescriptorDataset(
            val_paths,
            val_labels,
            transform_valid=self.transform_val,
            transform_fake=self.transform_val,
            descriptor=self.descriptor,
        )

        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

        return self.train_loader, self.val_loader

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_loader, total=len(self.train_loader)):
            # Extract HOG features from transformed images
            batch_features = np.array(inputs)
            labels = labels.numpy()

            # Fit scaler if first batch
            if not self.scaler_fitted:
                self.scaler.partial_fit(batch_features)

            # Scale features
            scaled_features = self.scaler.transform(batch_features)

            # Partial fit the model
            self.model.partial_fit(
                scaled_features,
                labels,
                classes=np.array([0, 1]),
            )

            # Calculate metrics
            predictions = self.model.predict(scaled_features)
            correct += (predictions == labels).sum()
            total += len(labels)

        self.scaler_fitted = True
        epoch_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return epoch_loss, accuracy

    def validate(self) -> tuple[float, float]:
        """Validate the model."""
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.val_loader, total=len(self.val_loader)):
            batch_features = np.array(inputs)
            labels = labels.numpy()

            # Scale features
            scaled_features = self.scaler.transform(batch_features)

            # Make predictions
            predictions = self.model.predict(scaled_features)
            # probas = self.model.predict_proba(scaled_features)

            # Calculate metrics
            correct += (predictions == labels).sum()
            total += len(labels)

        val_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return val_loss, accuracy

    def train(self, num_epochs=10):
        best_val_acc = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}:')

            # Regular training epoch
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Save best model
            if epoch > 0 and val_acc > best_val_acc:
                best_val_acc = val_acc
                joblib.dump((self.descriptor, self.model, self.scaler), self.save_path)

    def load(self):
        """Load the model."""
        self.descriptor, self.model, self.scaler = joblib.load(self.save_path)

    def predict(self, image_path):
        """Predict single image or group"""
        # Load and transform image
        if isinstance(image_path, str | Path):
            image = Image.open(image_path).convert('RGB')
            image = self.transform_val(image)
            # Extract features
            features = self.extract_features(image)
        else:
            images = [Image.open(im_p).convert('RGB') for im_p in image_path]
            images = [self.transform_val(image) for image in images]
            # Extract features
            features = np.concatenate([self.extract_features(image) for image in images])

        # Scale features and predict
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        pred = self.model.predict(scaled_features)[0]
        prob = self.model.predict_proba(scaled_features)[0][pred]

        return pred, prob


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
        skip_n=1
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
        self.skip_n = skip_n

    def val(self, general_model, decision, val_fullvid):
        # self.decision = decision
        # self.val_fullvid = val_fullvid
        self.general_model = general_model
        pass

    def set_seed(self, seed=0):
        """Set random seeds."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"Random seed set to {seed}")


    def train(self, datamodule, task_name):
        torch.cuda.empty_cache()
        self.set_seed(self.seed)
        pl.seed_everything(self.seed, workers=True)
        data_path = datamodule.data_dir
        split_path = datamodule.split_dir
        if isinstance(self.model, torch.nn.Module):
            save_path = f"bestmodels/{task_name}.pth"
            classifier = FrameClassifier(self.model, learning_rate=self.lr, save_path=save_path)
        else:
            save_path = f"bestmodels/{task_name}.joblib"
            valid_frames, nonvalid_frames = get_frames(f"{data_path}/origins/", splitfile=f"{split_path}/train.txt", skip_n=self.skip_n)
            train_frames = valid_frames + nonvalid_frames
            print(self.model)
            self.general_model.descriptor.fit(train_frames)

            classifier = DescriptorFrameClassifier(self.general_model.descriptor, learning_rate=self.lr, save_path=save_path)


        onlylegit = False
        legit_frames, fake_frames = get_image_paths(data_path, split_path, "train.txt", onlylegit, skip_n=self.skip_n)

        legit_frames_val, fake_frames_val = get_image_paths(data_path, split_path, "val.txt", onlylegit, skip_n=self.skip_n)

        val_paths = legit_frames_val + fake_frames_val
        val_labels = np.array([1] * len(legit_frames_val) + [0] * len(fake_frames_val))

        train_paths = legit_frames + fake_frames
        train_labels = np.array([1] * len(legit_frames) + [0] * len(fake_frames))
        print(train_paths[:5])
        print(len(train_labels), len(legit_frames))
        self.set_seed(self.seed)
        print(f"{datamodule.notransformations=}")
        train_loader, val_loader = classifier.prepare_data(
            train_paths=train_paths,
            train_labels=train_labels,
            val_paths=val_paths,
            val_labels=val_labels,
            num_workers=datamodule.num_workers,
            batch_size=datamodule.batch_size,
            notransformations=datamodule.notransformations)
        self.set_seed(self.seed)
        classifier.train(num_epochs=self.epochs * self.skip_n)  # * self.skip_n)
