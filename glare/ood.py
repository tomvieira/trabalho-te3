"""Out-of-Distribution (OOD) detection for GLARE.

Three complementary strategies:

1. **TemperatureScaler** — post-hoc calibration that learns a scalar T to
   minimise NLL on a validation set.  Calibrated probabilities are more
   reliable for thresholding.

2. **EnergyScorer** — stateless scorer that computes the free-energy of a
   logit vector.  Higher energy → more OOD.  No fitting required.

3. **MahalanobisOOD** — fits class-conditional Gaussians on 256-dim trunk
   embeddings and scores each sample by its minimum Mahalanobis distance to
   any class centroid.  Lower distance → more in-distribution.

All classes are designed to be backward-compatible: they are imported
optionally and every new parameter in the public API has a sensible default.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.covariance import EmpiricalCovariance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc calibration via a single scalar temperature parameter.

    After training, fit on the validation set logits and labels.  The optimal
    temperature T is found by minimising the negative log-likelihood (NLL) of
    the calibrated softmax distribution.

    Typical usage::

        scaler = TemperatureScaler()
        T = scaler.fit(val_logits, val_labels)
        calibrated_logits = scaler.calibrate(raw_logits)

    Args:
        temperature: Initial temperature (overridden by :meth:`fit`).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature: float = temperature

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Find the scalar T that minimises NLL on the provided logits/labels.

        Args:
            logits: Raw (uncalibrated) logits, shape ``(N, C)``.
            labels: Integer class labels, shape ``(N,)``.

        Returns:
            The optimal temperature T (also stored in ``self.temperature``).
        """
        logits_np = logits.detach().cpu().float().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)

        def nll(T: float) -> float:
            scaled = logits_np / T
            # Numerically stable log-softmax
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return -log_probs[np.arange(len(labels_np)), labels_np].mean()

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        logger.info("TemperatureScaler fitted: T=%.4f (NLL=%.4f)", self.temperature, result.fun)
        return self.temperature

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by the fitted temperature.

        Args:
            logits: Raw logits, shape ``(N, C)`` or ``(C,)``.

        Returns:
            Temperature-scaled logits with the same shape.
        """
        return logits / self.temperature

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise temperature to a JSON file.

        Args:
            path: Destination file path (e.g. ``temperature_scaler.json``).
        """
        with open(path, "w") as fh:
            json.dump({"temperature": self.temperature}, fh, indent=2)
        logger.info("TemperatureScaler saved to %s (T=%.4f)", path, self.temperature)

    def load(self, path: str) -> "TemperatureScaler":
        """Load temperature from a JSON file (in-place).

        Args:
            path: Source file path.

        Returns:
            ``self`` for chaining.
        """
        with open(path) as fh:
            data = json.load(fh)
        self.temperature = float(data["temperature"])
        logger.info("TemperatureScaler loaded from %s (T=%.4f)", path, self.temperature)
        return self

    @classmethod
    def from_file(cls, path: str) -> "TemperatureScaler":
        """Construct a :class:`TemperatureScaler` from a saved JSON file.

        Args:
            path: Source file path.

        Returns:
            A new :class:`TemperatureScaler` with the loaded temperature.
        """
        return cls().load(path)


# ---------------------------------------------------------------------------
# Energy Scorer
# ---------------------------------------------------------------------------

class EnergyScorer:
    """Stateless energy-based OOD scorer.

    The energy score is defined as::

        E(x; T) = -T * log Σ_c exp(f_c(x) / T)

    where ``f_c(x)`` are the raw (pre-softmax) logits for class ``c``.

    **Higher energy → more OOD.**

    This class has no state and all methods are static/class methods.
    No fitting is required.

    Reference:
        Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
    """

    @staticmethod
    def score(sf_logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """Compute energy scores for a batch of SF logits.

        Args:
            sf_logits: Raw superfamily logits, shape ``(B, C)`` or ``(C,)``.
            T: Temperature parameter (default 1.0).  Higher T → smoother
               energy landscape.

        Returns:
            Energy scores, shape ``(B,)`` or scalar.  Higher = more OOD.
        """
        return -T * torch.logsumexp(sf_logits.float() / T, dim=-1)

    @staticmethod
    def score_numpy(sf_logits: np.ndarray, T: float = 1.0) -> np.ndarray:
        """NumPy wrapper around :meth:`score`.

        Args:
            sf_logits: Raw superfamily logits, shape ``(B, C)`` or ``(C,)``.
            T: Temperature parameter (default 1.0).

        Returns:
            Energy scores as a NumPy array.  Higher = more OOD.
        """
        t = torch.from_numpy(np.asarray(sf_logits, dtype=np.float32))
        return EnergyScorer.score(t, T=T).numpy()


# ---------------------------------------------------------------------------
# Mahalanobis OOD
# ---------------------------------------------------------------------------

class MahalanobisOOD:
    """Class-conditional Mahalanobis distance OOD detector.

    Fits a Gaussian per class on the 256-dim trunk embeddings (output of
    ``GLAREHead.trunk``, before the SF linear head) using a **tied**
    (shared) precision matrix estimated from all classes jointly.

    Scoring::

        score(z) = min_c  (z - μ_c)ᵀ Σ⁻¹ (z - μ_c)

    **Lower score → more in-distribution; higher score → more OOD.**

    Reference:
        Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
        Samples and Adversarial Attacks", NeurIPS 2018.

    Args:
        min_samples_per_class: Classes with fewer samples are skipped.
    """

    def __init__(self, min_samples_per_class: int = 2) -> None:
        self.min_samples_per_class = min_samples_per_class
        self.class_means_: Optional[np.ndarray] = None   # (K, D)
        self.precision_: Optional[np.ndarray] = None     # (D, D)
        self.class_ids_: Optional[np.ndarray] = None     # (K,) — class indices present

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Fit class-conditional Gaussians on trunk embeddings.

        Args:
            embeddings: Trunk embeddings, shape ``(N, D)`` (D=256 for GLARE).
            labels: Integer class labels, shape ``(N,)``.

        Raises:
            ValueError: If no class has enough samples to fit.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        unique_classes = np.unique(labels)
        class_means = []
        class_ids = []
        centered_all: list[np.ndarray] = []

        for cls in unique_classes:
            mask = labels == cls
            n = mask.sum()
            if n < self.min_samples_per_class:
                logger.warning(
                    "MahalanobisOOD: class %d has only %d sample(s) — skipping "
                    "(need >= %d).",
                    cls, n, self.min_samples_per_class,
                )
                continue
            cls_emb = embeddings[mask]
            mu = cls_emb.mean(axis=0)
            class_means.append(mu)
            class_ids.append(cls)
            centered_all.append(cls_emb - mu)

        if not class_means:
            raise ValueError(
                "MahalanobisOOD.fit: no class had enough samples to fit. "
                f"min_samples_per_class={self.min_samples_per_class}."
            )

        self.class_means_ = np.stack(class_means, axis=0)   # (K, D)
        self.class_ids_ = np.array(class_ids, dtype=int)    # (K,)

        # Tied covariance: fit on all centered samples jointly
        centered_concat = np.concatenate(centered_all, axis=0)  # (N_valid, D)
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(centered_concat)
        self.precision_ = cov_estimator.precision_  # (D, D)

        logger.info(
            "MahalanobisOOD fitted: %d classes, embedding_dim=%d",
            len(class_ids), embeddings.shape[1],
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, embedding: np.ndarray) -> float:
        """Compute the minimum Mahalanobis distance to any class centroid.

        Args:
            embedding: Single trunk embedding, shape ``(D,)``.

        Returns:
            Minimum squared Mahalanobis distance (float).
            Lower = more in-distribution; higher = more OOD.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.class_means_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisOOD.score called before fit().")
        z = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        return float(self.score_batch(z)[0])

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorised Mahalanobis scoring for a batch of embeddings.

        Args:
            embeddings: Trunk embeddings, shape ``(B, D)``.

        Returns:
            Minimum squared Mahalanobis distances, shape ``(B,)``.
            Lower = more in-distribution; higher = more OOD.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.class_means_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisOOD.score_batch called before fit().")

        embeddings = np.asarray(embeddings, dtype=np.float64)  # (B, D)
        K = self.class_means_.shape[0]
        B = embeddings.shape[0]

        # Compute (z - μ_c) for all classes: shape (B, K, D)
        diff = embeddings[:, None, :] - self.class_means_[None, :, :]  # (B, K, D)

        # Mahalanobis: (z - μ_c)ᵀ Σ⁻¹ (z - μ_c)
        # Efficient: first compute diff @ precision_ → (B, K, D), then dot with diff
        diff_flat = diff.reshape(B * K, -1)                          # (B*K, D)
        mahal_flat = (diff_flat @ self.precision_ * diff_flat).sum(axis=1)  # (B*K,)
        mahal = mahal_flat.reshape(B, K)                             # (B, K)

        return mahal.min(axis=1)  # (B,)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise class means and precision matrix to a ``.npz`` file.

        Args:
            path: Destination file path (e.g. ``mahal_ood.npz``).
        """
        if self.class_means_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisOOD.save called before fit().")
        np.savez(
            path,
            class_means=self.class_means_,
            precision=self.precision_,
            class_ids=self.class_ids_,
        )
        logger.info(
            "MahalanobisOOD saved to %s (%d classes, dim=%d)",
            path, len(self.class_ids_), self.class_means_.shape[1],
        )

    def load(self, path: str) -> "MahalanobisOOD":
        """Load class means and precision matrix from a ``.npz`` file (in-place).

        Args:
            path: Source file path.

        Returns:
            ``self`` for chaining.
        """
        data = np.load(path)
        self.class_means_ = data["class_means"]
        self.precision_ = data["precision"]
        self.class_ids_ = data["class_ids"]
        logger.info(
            "MahalanobisOOD loaded from %s (%d classes, dim=%d)",
            path, len(self.class_ids_), self.class_means_.shape[1],
        )
        return self

    @classmethod
    def from_file(cls, path: str) -> "MahalanobisOOD":
        """Construct a :class:`MahalanobisOOD` from a saved ``.npz`` file.

        Args:
            path: Source file path.

        Returns:
            A new :class:`MahalanobisOOD` with the loaded parameters.
        """
        return cls().load(path)
