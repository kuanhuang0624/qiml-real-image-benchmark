"""Portable train-only PCA angle interface for examples and new experiments.

The original experiment's cache builder is fit_shared_interfaces.py. This
adapter retains PCA centering explicitly for new runs; it does not replace
any historical cache or claim bitwise reproduction of randomized PCA.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class AngleInterface:
    width: int = 8
    seed: int = 20260812

    def fit(self, features: np.ndarray, *, split: str = 'train'):
        if split != 'train': raise ValueError('PCA/scaling may only be fitted on training data')
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all(): raise ValueError('Expected finite feature matrix')
        self.scaler = StandardScaler().fit(values)
        standardized = self.scaler.transform(values)
        self.pca = PCA(self.width, svd_solver='randomized', random_state=self.seed).fit(standardized)
        self.quantiles = np.quantile(np.abs(self.pca.transform(standardized)), .99, axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = self.pca.transform(self.scaler.transform(np.asarray(features, dtype=np.float32)))
        return (np.pi * np.clip(values / np.maximum(self.quantiles, 1e-8), -1, 1)).astype(np.float32)

    def arrays(self) -> dict:
        return {'mean': self.scaler.mean_, 'std': self.scaler.scale_, 'pca_mean': self.pca.mean_,
                'components': self.pca.components_, 'quantiles': self.quantiles}
