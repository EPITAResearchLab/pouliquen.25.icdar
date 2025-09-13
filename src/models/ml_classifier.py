from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms as T

# class DescriptorClassifier:
#     def __init__(self, model, load_path: str | None):
#         self.save_path = load_path
#         self.descriptor = descriptor

#         if load_path is not None:
#             self.model, self.scaler, self.descriptor = joblib.load(self.save_path)
#         else:
#             self.model = model
#             self.scaler = StandardScaler()

#     def predict(self, image_path):
#         """Predict single image or group"""
#         # Load and transform image
#         if isinstance(image_path, str | Path):
#             image = Image.open(image_path).convert('RGB')
#             image = self.transform_val(image)
#             # Extract features
#             features = self.extract_features(image)
#         else:
#             images = [Image.open(im_p).convert('RGB') for im_p in image_path]
#             images = [self.transform_val(image) for image in images]
#             # Extract features
#             features = np.concatenate([self.extract_features(image) for image in images])

#         # Scale features and predict
#         scaled_features = self.scaler.transform(features.reshape(1, -1))
#         return self.model.predict(scaled_features)[0]

#     # def predict_batch(self,
#     #                   image_paths,
#     #                   batch_size=64,
#     #                   ) -> tuple[np.ndarray, np.ndarray]:
#     #     """Predict multiple images in batches."""
#     #     all_preds = []
#     #     all_probs = []

#     #     # Create a simple dataset and dataloader for batch processing
#     #     dataset_class = (DescriptorDatasetWithHardNegatives
#     #                      if isinstance(image_paths[0], (str, Path))
#     #                      else GroupDescriptorFrameDataset)
#     #     dataset = dataset_class(
#     #         image_paths,
#     #         [0] * len(image_paths),  # Dummy labels
#     #         transform_valid=self.transform_val,
#     #         transform_fake=self.transform_val,
#     #         descriptor=self.descriptor,
#     #     )
#     #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     #     for inputs, _ in dataloader:
#     #         batch_features = np.array(inputs)
#     #         # Get predictions and probabilities
#     #         scaled_features = self.scaler.transform(batch_features)
#     #         probas = self.model.predict_proba(scaled_features)
#     #         preds = self.model.predict(scaled_features)
#     #         all_preds.extend(preds)
#     #         all_probs.extend(probas)
#     #     return all_preds, all_probs
log = logging.getLogger(__name__)


class MLClassification:
    """Classification Model."""

    MIN_SIZE = 3
    def __init__(self, model: any, descriptor: any, model_path="") -> None:
        """Initialize."""
        self.model = model

        self.descriptor = descriptor
        print(f"{self.model=}")
        print(f"{self.descriptor=}")
        self.scaler = None
        if model_path is not None and len(model_path) != 0:
            log.info("loading checkpoint from path %s", model_path)
            self.descriptor, self.model, self.scaler = joblib.load(model_path)
        # self.transform_val = transform_val
        self.reset()

    def reset(self) -> None:
        """Reset all vars."""
        self.preds = np.array([])

    def predict(self, image):
        """Predict single image."""
        # Load and transform image
        # if isinstance(image_path, (str, Path)):
        #     image = Image.open(image_path).convert('RGB')
        #     image = self.transform_val(image)
        #     # Extract features
        #     features = self.extract_descriptor(image)
        # else:
        #     # image = self.transform_val(image_path)
        features = self.descriptor.extract_descriptor(np.array(image))

        # Scale features and predict
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(scaled_features)[0]

    def apply(self, img_t: np.array) -> float:
        """Apply a new tensor and returns the class.

        Returns:
            Classification class
        """
        pred = self.predict(img_t)

        self.preds = np.append(self.preds, pred)
        if self.preds.size > self.MIN_SIZE:
            return self.preds.mean()
        return 0

class MLClassificationWithLabels:
    """Classification Model suign pseudo labels."""

    MIN_SIZE = 3
    def __init__(self, model: any, descriptor: any, model_path="") -> None:
        """Initialize."""
        self.model = model

        self.descriptor = descriptor
        print(f"{self.model=}")
        print(f"{self.descriptor=}")
        self.scaler = None
        if model_path is not None and len(model_path) != 0:
            log.info("loading checkpoint from path %s", model_path)
            self.descriptor, self.model, self.scaler = joblib.load(model_path)
        # self.transform_val = transform_val
        self.reset()

    def reset(self) -> None:
        """Reset all vars."""
        self.preds = np.array([])

    def predict(self, image):
        """Predict single image."""
        # Load and transform image
        # if isinstance(image_path, (str, Path)):
        #     image = Image.open(image_path).convert('RGB')
        #     image = self.transform_val(image)
        #     # Extract features
        #     features = self.extract_descriptor(image)
        # else:
        #     # image = self.transform_val(image_path)
        features = self.descriptor.extract_descriptor(np.array(image))

        # Scale features and predict
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(scaled_features)[0]

    def apply(self, img_t: np.array) -> float:
        """Apply a new tensor and returns the class.

        Returns:
            Classification class
        """
        pred = self.predict(img_t)

        self.preds = np.append(self.preds, pred)
        if self.preds.size > self.MIN_SIZE:
            return self.preds.mean()
        return 0