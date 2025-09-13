from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import tqdm
from joblib import dump, load
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AbstractDescriptorExtractor(ABC):
    """Asbtract class for features extraction using a descriptor."""

    @abstractmethod
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a frame."""

    @abstractmethod
    def get_descriptor_size(self) -> int:
        """Return the size of the features from the descriptor."""


class ORBExtractor(AbstractDescriptorExtractor):
    def __init__(self,
                 score_type: int = cv2.ORB_FAST_SCORE,
                 **kwargs: dict | None) -> None:
        """Features extraction using ORB."""
        self.scoreType = score_type
        self.kwargs = kwargs
        self.orb = cv2.ORB_create(
            scoreType=score_type,
            **kwargs,
        )

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.orb.detectAndCompute(gray, None)

        return descriptors if descriptors is not None else np.array([])

    def get_descriptor_size(self):
        return 32

    def __getstate__(self):
        # Return the state of the object, excluding the SIFT object
        state = self.__dict__.copy()
        del state['orb']
        return state

    def __setstate__(self, state):
        # Restore the state and recreate the SIFT object
        self.__dict__.update(state)
        self.orb = cv2.ORB_create(scoreType=self.score_type,
            **self.kwargs)


class SIFTExtractor(AbstractDescriptorExtractor):
    def __init__(self,
                 nfeatures: int = 2000,
                 **kwargs: dict | None) -> None:
        """Features extraction using SIFT."""
        self.nfeatures = nfeatures
        self.kwargs = kwargs
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            **kwargs,
        )

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors if descriptors is not None else np.array([])

    def get_descriptor_size(self):
        return 128

    def __getstate__(self):
        # Return the state of the object, excluding the SIFT object
        state = self.__dict__.copy()
        del state['sift']
        return state

    def __setstate__(self, state):
        # Restore the state and recreate the SIFT object
        self.__dict__.update(state)
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures, **self.kwargs)


class SIFTColorExtractor(AbstractDescriptorExtractor):
    def __init__(self,
                 nfeatures: int = 2000,
                 **kwargs: dict | None) -> None:
        """Features extraction using SIFT."""
        self.nfeatures = nfeatures
        self.kwargs = kwargs
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            **kwargs,
        )

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        descriptors_list = []
        for i in range(3):
            _, descriptors_tmp = self.sift.detectAndCompute(frame[..., i], None)
            if descriptors_tmp is not None:
                descriptors_list.append(descriptors_tmp)

        return np.vstack(descriptors_list) if descriptors_list else np.array([])

    def get_descriptor_size(self):
        return 128

    def __getstate__(self):
        # Return the state of the object, excluding the SIFT object
        state = self.__dict__.copy()
        del state['sift']
        return state

    def __setstate__(self, state):
        # Restore the state and recreate the SIFT object
        self.__dict__.update(state)
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures, **self.kwargs)


class HOGExtractor(AbstractDescriptorExtractor):
    def __init__(self,
                 win_size: tuple[int, int] = (64, 64),
                 block_size: tuple[int, int] = (16, 16),
                 block_stride: tuple[int, int] = (8, 8),
                 cell_size: tuple[int, int] = (8, 8),
                 nbins: int = 9) -> None:
        """Features extraction using HOG."""
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        self._descriptor_size = None

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.hog.winSize)
        features = self.hog.compute(resized)
        if self._descriptor_size is None:
            self._descriptor_size = features.shape[0]
        return features.flatten()

    def get_descriptor_size(self) -> int:
        if self._descriptor_size is None:
            msg = "Extract features from at least one frame to set the descriptor size"
            raise ValueError(msg)
        return self._descriptor_size

    def __getstate__(self):
        # Return the state of the object, excluding the SIFT object
        state = self.__dict__.copy()
        del state['hog']
        return state

    def __setstate__(self, state):
        # Restore the state and recreate the SIFT object
        self.__dict__.update(state)
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride,
                                     self.cell_size, self.nbins)


class ColorHistogramExtractor(AbstractDescriptorExtractor):
    def __init__(self, bins: int = 32, channels: list[int] | None = None) -> None:
        """Features extraction using color histograms."""
        self.bins = bins
        if channels is None:
            channels = [0, 1, 2]
        self.channels = channels

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        features = []
        for channel in self.channels:
            hist = cv2.calcHist([frame], [channel], None, [self.bins], [0, 256])
            features.extend(hist.flatten())
        return np.array(features)

    def get_descriptor_size(self) -> int:
        return self.bins * len(self.channels)


class FrameDescriptorExtractorGeneral:
    def __init__(self,
                 n_clusters: int = 64,
                 pca_components: int = 256,
                 descriptor_extractor: AbstractDescriptorExtractor | None = None,
                 enable_vlad: bool = True,
                 ) -> None:
        """Initialize the frame descriptor extractor.

        Args:
            n_clusters (int): Number of visual words for VLAD
            pca_components (int): Final descriptor size after PCA
            orb_params (dict): Parameters for ORB detector (optional)
        """
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.enable_vlad = enable_vlad
        print("VLAD", self.enable_vlad)

        # Initialize ORB detector as default.
        orb_params = {
            "scoreType": cv2.ORB_FAST_SCORE,
            "edgeThreshold": 5,
            "patchSize": 5,
            "fastThreshold": 5,
            "nfeatures": 500,  # Limit number of features for consistent sizing
        }
        self.descriptor = descriptor_extractor or ORBExtractor(**orb_params)

        # Initialize preprocessing, clustering and PCA
        self.scaler = StandardScaler()
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)

        self.pca = PCA(n_components=pca_components, random_state=42)

        # Storage for learned parameters
        self.centroids = None
        self.is_fitted = False

    def extract_frame_features(self, frame: str | Path | np.ndarray) -> np.ndarray:
        """Extract features from a single frame"""
        if isinstance(frame, (str, Path)):
            frame = cv2.imread(str(frame))
        return self.descriptor.extract_features(frame)

    def compute_vlad(self, descriptors: np.ndarray) -> np.ndarray:
        """Compute VLAD descriptor for a set of local descriptors."""
        # Scale the descriptors using the fitted scaler
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters * self.descriptor.get_descriptor_size())

        descriptors = self.scaler.transform(descriptors)

        # Compute distances to centroids
        distances = np.sqrt(((descriptors[:, np.newaxis, :] -
                            self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

        # Hard assignment to nearest cluster
        assignments = distances.argmin(axis=1)

        # Initialize VLAD vector
        vlad = np.zeros((self.n_clusters, descriptors.shape[1]))

        # Accumulate residuals for each cluster
        for i in range(self.n_clusters):
            if np.sum(assignments == i) > 0:
                vlad[i] = np.sum(descriptors[assignments == i] - self.centroids[i], axis=0)

        # Flatten and normalize
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))  # Power normalization
        vlad /= (np.sqrt(np.sum(vlad ** 2)) + 1e-12)  # L2 normalization

        return vlad

    def fit(self, training_frames: str | Path | np.ndarray) -> None:
        """Learn VLAD vocabulary and PCA from training frames.

        Args:
            training_frames: List of frame paths or numpy arrays
        """
        print("Extracting features from training frames...")
        frame_features = []  # Store features for each frame
        all_features = []    # Store all feature vectors for scaler/kmeans

        # Extract features once and store them
        for frame in tqdm.tqdm(training_frames):
            features = self.extract_frame_features(frame)
            if features.size > 0:
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                frame_features.append(features)
                all_features.append(features)

        if not all_features:
            raise ValueError("No valid features extracted from training frames")

        all_features = np.vstack(all_features)
        print(f"Fitting scaler on {len(all_features)} feature vectors...")
        all_features = self.scaler.fit_transform(all_features)

        if self.enable_vlad:
            print("Training KMeans for VLAD...")
            self.kmeans.fit(all_features)
            self.centroids = self.kmeans.cluster_centers_

            # Prepare data for PCA using stored features
            print("Computing descriptors for PCA training...")
            pca_vectors = []
            for features in tqdm.tqdm(frame_features):
                vector = self.compute_vlad(features)
                pca_vectors.append(vector)

            print("Fitting PCA...")
            self.pca.fit(pca_vectors)
        self.is_fitted = True
        print("Training complete!")

    def extract_descriptor(self, frame: str | np.ndarray) -> np.ndarray:
        """Extract fixed-size descriptor for a single frame.

        Args:
            frame: Frame path or numpy array

        Returns:
            np.ndarray: Fixed-size frame descriptor

        Raises:
            ValueError: If the model has not been fitted yet
        """
        if not self.is_fitted:
            msg = "Must fit the model first!"
            raise ValueError(msg)

        descriptors = self.extract_frame_features(frame)

        if self.enable_vlad:
            # Compute VLAD
            vlad = self.compute_vlad(descriptors)
            # Reduce dimensionality with PCA
            descriptors = self.pca.transform([vlad])[0]
        # else:
        #     descriptors = descriptors.reshape(1, -1)
        return descriptors

    def extract_descriptors(self, frames):
        """Extract descriptors for multiple frames"""
        return np.array([self.extract_descriptor(frame) for frame in frames])

    def save(self, path: str | Path) -> None:
        """Save the trained model using joblib.

        Args:
            path: Path to save the model!

        Raises:
            ValueError: If the model has not been fitted yet :'(
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model!")

        model_data = {
            "n_clusters": self.n_clusters,
            "pca_components": self.pca_components,
            "enable_vlad": self.enable_vlad,
            "descriptor": self.descriptor,
            "scaler": self.scaler,
            "kmeans": self.kmeans,
            "pca": self.pca,
            "centroids": self.centroids,
            "is_fitted": self.is_fitted,
        }
        dump(model_data, path)

    @classmethod
    def load(cls, path: str | Path) -> FrameDescriptorExtractorGeneral:
        data = load(path)

        # Create new instance
        model = cls(
            n_clusters=data['n_clusters'],
            pca_components=data['pca_components'],
            descriptor_extractor=data['descriptor'],
            enable_vlad=data["enable_vlad"],
        )

        # load model
        model.scaler = data['scaler']
        model.kmeans = data['kmeans']
        model.pca = data['pca']
        model.centroids = data['centroids']
        model.is_fitted = data['is_fitted']
        model.enable_vlad = data["enable_vlad"]

        return model
