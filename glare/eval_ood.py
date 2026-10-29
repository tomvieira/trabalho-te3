"""OOD evaluation pipeline for GLARE.

Evaluates out-of-distribution detection quality on the held-out test sets
(panteon_random_test and repbase_random_test) by treating correctly classified
sequences as in-distribution (ID) and misclassified sequences as pseudo-OOD.

Also supports true OOD evaluation against real non-TE FASTA sequences via
:func:`evaluate_ood_vs_fasta`.

Usage::

    python -m glare.glare.eval_ood \\
        --model-path glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/best_model.pt \\
        --config glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment.yaml \\
        --output-dir glare_ood_eval_results/ \\
        --temperature-scaler glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/temperature_scaler.json \\
        --mahal-detector glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/mahal_ood.npz

    # True OOD vs real non-TE FASTA:
    python -m glare.glare.eval_ood \\
        --model-path glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/best_model.pt \\
        --config glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment.yaml \\
        --output-dir glare_ood_fasta_eval_results \\
        --temperature-scaler glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/temperature_scaler.json \\
        --mahal-detector glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/mahal_ood.npz \\
        --eval-mode fasta \\
        --id-fasta glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta \\
        --ood-fasta glare/data/non-TE/arabidopsis_high_confidence_cds.fa \\
        --max-sequences 10000
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA subsampling helpers
# ---------------------------------------------------------------------------

def _count_fasta_sequences(fasta_path: str) -> int:
    """Count the number of sequences in a FASTA file by counting '>' lines."""
    count = 0
    with open(fasta_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def _subsample_fasta(fasta_path: str, n: int, seed: int = 42) -> Path:
    """Write a randomly subsampled FASTA to a temp file and return its path.

    Reads all sequence headers first to determine which indices to keep, then
    makes a second pass to write only those sequences.  Uses ``random.seed``
    for reproducibility.

    Args:
        fasta_path: Path to the source FASTA file.
        n: Number of sequences to sample.
        seed: Random seed for reproducibility.

    Returns:
        Path to a temporary FASTA file containing the subsampled sequences.
        The caller is responsible for deleting this file after use.
    """
    from Bio import SeqIO

    # First pass: collect all record IDs
    all_ids: list[str] = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        all_ids.append(rec.id)

    total = len(all_ids)
    if n >= total:
        logger.info(
            "Subsampling requested %d but file has only %d sequences — using all.",
            n, total,
        )
        return Path(fasta_path)

    rng = random.Random(seed)
    selected_ids = set(rng.sample(all_ids, n))
    logger.info(
        "Subsampling %d / %d sequences from %s (seed=%d).",
        n, total, fasta_path, seed,
    )

    # Second pass: write selected records to a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fa", delete=False, prefix="glare_subsample_"
    )
    tmp_path = Path(tmp.name)
    written = 0
    with tmp:
        for rec in SeqIO.parse(fasta_path, "fasta"):
            if rec.id in selected_ids:
                tmp.write(f">{rec.id}\n{str(rec.seq)}\n")
                written += 1
                if written >= n:
                    break

    logger.info("Wrote %d sequences to temp file %s.", written, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# OOD metric helpers
# ---------------------------------------------------------------------------

def _default_fpr_targets() -> list[float]:
    """Return the default FPR targets used for threshold reporting."""
    return [0.05, 0.01, 0.005, 0.001]


def _format_fpr_target_key(target: float) -> str:
    """Format an FPR target as a stable column/key suffix.

    Examples: ``0.05`` → ``"5pct"``, ``0.005`` → ``"0_5pct"``.
    """
    pct = target * 100.0
    return f"{pct:g}".replace(".", "_") + "pct"


def _nan_ood_metrics(fpr_targets: Optional[list[float]] = None) -> dict:
    """Return a metrics dictionary filled with NaNs for undefined scorers."""
    if fpr_targets is None:
        fpr_targets = [0.05]

    metrics = {
        "auroc": float("nan"),
        "aupr": float("nan"),
        "fpr_at_tpr95": float("nan"),
        "threshold_at_fpr5": float("nan"),
        "fpr_at_threshold": float("nan"),
        "tpr_at_threshold": float("nan"),
        "thresholds_at_fpr": {},
    }
    for target in fpr_targets:
        target_float = float(target)
        key = _format_fpr_target_key(target_float)
        metrics["thresholds_at_fpr"][str(target_float)] = {
            "threshold": float("nan"),
            "fpr_at_threshold": float("nan"),
            "tpr_at_threshold": float("nan"),
        }
        metrics[f"threshold_at_fpr_{key}"] = float("nan")
        metrics[f"fpr_at_threshold_at_fpr_{key}"] = float("nan")
        metrics[f"tpr_at_threshold_at_fpr_{key}"] = float("nan")
    return metrics


def _compute_ood_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    scorer_name: str,
    fpr_targets: Optional[list[float]] = None,
) -> dict:
    """Compute AUROC, AUPR, FPR@TPR95, and thresholds at FPR targets.

    Args:
        y_true: Binary array where 1 = OOD (misclassified), 0 = ID (correct).
        scores: OOD scores (higher = more OOD).
        scorer_name: Name of the scorer (for logging).
        fpr_targets: FPR targets for threshold reporting.  Defaults to ``[0.05]``
            to preserve the existing FPR=5% fields for legacy evaluations.

    Returns:
        Dict with keys: auroc, aupr, fpr_at_tpr95, threshold_at_fpr5,
        fpr_at_threshold, tpr_at_threshold, plus per-target threshold fields.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    if fpr_targets is None:
        fpr_targets = [0.05]

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        logger.warning(
            "Scorer %s: all labels are the same (%d OOD, %d ID) — metrics undefined.",
            scorer_name, int(y_true.sum()), int((1 - y_true).sum()),
        )
        return _nan_ood_metrics(fpr_targets)

    auroc = float(roc_auc_score(y_true, scores))
    aupr = float(average_precision_score(y_true, scores))

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # FPR@TPR95: find FPR when TPR >= 0.95
    idx95 = np.searchsorted(tpr, 0.95)
    fpr_at_tpr95 = float(fpr[min(idx95, len(fpr) - 1)])

    thresholds_at_fpr: dict[str, dict] = {}
    per_target_flat: dict[str, float] = {}
    for target in fpr_targets:
        target_float = float(target)
        # Choose the operating point with the largest observed FPR that does not
        # exceed the requested target.  This yields the highest TPR available
        # under the target FPR constraint and avoids returning the initial
        # infinite ROC threshold when zero-FPR finite thresholds exist.
        idx = int(np.searchsorted(fpr, target_float, side="right") - 1)
        idx = max(0, min(idx, len(thresholds) - 1))
        tau = float(thresholds[idx])
        fpr_at_tau = float(fpr[idx])
        tpr_at_tau = float(tpr[idx])

        thresholds_at_fpr[str(target_float)] = {
            "threshold": tau,
            "fpr_at_threshold": fpr_at_tau,
            "tpr_at_threshold": tpr_at_tau,
        }
        key = _format_fpr_target_key(target_float)
        per_target_flat[f"threshold_at_fpr_{key}"] = tau
        per_target_flat[f"fpr_at_threshold_at_fpr_{key}"] = fpr_at_tau
        per_target_flat[f"tpr_at_threshold_at_fpr_{key}"] = tpr_at_tau

    fpr5_entry = thresholds_at_fpr.get("0.05")
    if fpr5_entry is None:
        fpr5_entry = next(iter(thresholds_at_fpr.values()))

    metrics = {
        "auroc": auroc,
        "aupr": aupr,
        "fpr_at_tpr95": fpr_at_tpr95,
        # Backward-compatible aliases for the 5% FPR operating point when it is
        # requested; otherwise these point at the first requested FPR target.
        "threshold_at_fpr5": fpr5_entry["threshold"],
        "fpr_at_threshold": fpr5_entry["fpr_at_threshold"],
        "tpr_at_threshold": fpr5_entry["tpr_at_threshold"],
        "thresholds_at_fpr": thresholds_at_fpr,
    }
    metrics.update(per_target_flat)
    return metrics


def _display_threshold_for_scorer(scorer_name: str, raw_threshold: float) -> float:
    """Convert an OOD-score threshold to the original scorer scale for output."""
    return -raw_threshold if scorer_name in ("sf_conf", "hier_conf") else raw_threshold


def _display_threshold_metrics_for_scorer(
    scorer_name: str,
    metrics: dict,
    fpr_targets: list[float],
) -> dict:
    """Return threshold metrics with confidence scorers restored to conf scale."""
    display: dict = {}
    for target in fpr_targets:
        target_float = float(target)
        target_key = str(target_float)
        suffix = _format_fpr_target_key(target_float)
        entry = metrics.get("thresholds_at_fpr", {}).get(target_key, {})
        raw_tau = entry.get("threshold", float("nan"))
        display_tau = _display_threshold_for_scorer(scorer_name, raw_tau)
        display[target_key] = {
            "threshold": display_tau,
            "fpr_at_threshold": entry.get("fpr_at_threshold", float("nan")),
            "tpr_at_threshold": entry.get("tpr_at_threshold", float("nan")),
        }
        display[f"threshold_at_fpr_{suffix}"] = display_tau
        display[f"fpr_at_threshold_at_fpr_{suffix}"] = entry.get("fpr_at_threshold", float("nan"))
        display[f"tpr_at_threshold_at_fpr_{suffix}"] = entry.get("tpr_at_threshold", float("nan"))
    return display


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_roc_curves(
    results_per_scorer: dict[str, dict],
    test_set_name: str,
    output_dir: Path,
) -> None:
    """Plot ROC curves for all scorers on one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"sf_conf": "#1f77b4", "hier_conf": "#ff7f0e", "energy_score": "#2ca02c", "mahal_score": "#d62728"}
    labels_map = {
        "sf_conf": "SF Conf (negated)",
        "hier_conf": "Hier Conf (negated)",
        "energy_score": "Energy Score",
        "mahal_score": "Mahal Score",
    }

    from sklearn.metrics import roc_curve

    for scorer, metrics in results_per_scorer.items():
        if np.isnan(metrics.get("auroc", float("nan"))):
            continue
        y_true = metrics["_y_true"]
        scores = metrics["_scores"]
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = metrics["auroc"]
        ax.plot(
            fpr, tpr,
            color=colors.get(scorer, None),
            label=f"{labels_map.get(scorer, scorer)} (AUROC={auroc:.3f})",
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {test_set_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    out_path = output_dir / f"roc_curves_{test_set_name}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", out_path)


def _plot_score_distributions(
    results_per_scorer: dict[str, dict],
    test_set_name: str,
    output_dir: Path,
) -> None:
    """Plot score distribution histograms (ID vs OOD) for each scorer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer_names = list(results_per_scorer.keys())
    n = len(scorer_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels_map = {
        "sf_conf": "SF Conf (negated)",
        "hier_conf": "Hier Conf (negated)",
        "energy_score": "Energy Score",
        "mahal_score": "Mahal Score",
    }

    for ax, scorer in zip(axes, scorer_names):
        metrics = results_per_scorer[scorer]
        y_true = metrics["_y_true"]
        scores = metrics["_scores"]

        id_scores = scores[y_true == 0]
        ood_scores = scores[y_true == 1]

        bins = 50
        ax.hist(id_scores, bins=bins, alpha=0.6, color="#1f77b4", label="ID (correct)", density=True)
        ax.hist(ood_scores, bins=bins, alpha=0.6, color="#d62728", label="OOD (incorrect)", density=True)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(labels_map.get(scorer, scorer), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score Distributions — {test_set_name}", fontsize=13)
    out_path = output_dir / f"score_distributions_{test_set_name}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("Score distributions saved to %s", out_path)


def _plot_calibration(
    sf_conf_values: np.ndarray,
    correct_values: np.ndarray,
    test_set_name: str,
    output_dir: Path,
    label: str = "sf_conf",
) -> None:
    """Plot reliability diagram (calibration curve) for sf_conf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if len(sf_conf_values) < 10:
        logger.warning("Too few samples for calibration plot (%d), skipping.", len(sf_conf_values))
        return

    n_bins = 10
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            correct_values.astype(int),
            sf_conf_values,
            n_bins=n_bins,
            strategy="uniform",
        )
    except Exception as exc:
        logger.warning("Calibration curve failed for %s: %s", label, exc)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title(f"Calibration — {test_set_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    out_path = output_dir / f"calibration_{test_set_name}_{label}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("Calibration plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Inference on labeled records
# ---------------------------------------------------------------------------

def _run_inference_on_records(
    records: list[dict],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    temperature_scaler,
    mahal_detector,
    use_hierarchical_confidence: bool,
    idx2superfamily: dict | None = None,
    sf_to_order_fn=None,
    order_sf_indices: dict | None = None,
) -> list[dict]:
    """Run inference on a list of labeled records.

    Each record must have keys: ``sequence``, ``superfamily``.

    Args:
        idx2superfamily: Mapping from integer index to superfamily label string.
            If ``None``, falls back to the module-level PanTEon 30-SF mapping
            (kept for backward compatibility, but callers should always pass the
            label-space-correct dict).
        sf_to_order_fn: Callable ``(sf_label: str) -> str`` mapping a superfamily
            label to its order.  If ``None``, falls back to the module-level
            PanTEon ``sf_to_order``.
        order_sf_indices: Mapping from order name to list of SF indices belonging
            to that order.  Used for hierarchical confidence computation.  If
            ``None``, falls back to the module-level PanTEon ``ORDER_SF_INDICES``
            from ``predict.py`` (kept for backward compatibility, but callers
            should always pass the label-space-correct dict).

    Returns a list of dicts with per-sequence results.
    """
    from torch.utils.data import DataLoader, Dataset
    from .constants import IDX2SUPERFAMILY as _DEFAULT_IDX2SF, sf_to_order as _default_sf_to_order
    from .data import DNABERT2Tokenizer
    from .ood import EnergyScorer
    from .predict import ORDER_SF_INDICES as _DEFAULT_ORDER_SF_INDICES

    # Use caller-supplied label-space mappings when available
    if idx2superfamily is None:
        idx2superfamily = _DEFAULT_IDX2SF
    if sf_to_order_fn is None:
        sf_to_order_fn = _default_sf_to_order
    if order_sf_indices is None:
        order_sf_indices = _DEFAULT_ORDER_SF_INDICES

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    class _LabeledDataset(Dataset):
        def __init__(self, recs, tok, max_len):
            self.recs = recs
            self.tok = tok
            self.max_len = max_len
            self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            rec = self.recs[idx]
            seq = rec["sequence"]
            if isinstance(self.tok, DNABERT2Tokenizer) and len(seq) > self.max_len:
                seq = seq[: self.max_len]
            enc = self.tok(
                seq,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                pad_to_multiple_of=128,
            )
            input_ids = enc["input_ids"].squeeze(0)
            attn_mask = (
                enc["attention_mask"].squeeze(0)
                if "attention_mask" in enc
                else (input_ids != self.pad_id).long()
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "true_superfamily": rec.get("superfamily", ""),
                "seq_id": rec.get("id", str(idx)),
            }

    def _collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "true_superfamily": [b["true_superfamily"] for b in batch],
            "seq_id": [b["seq_id"] for b in batch],
        }

    dataset = _LabeledDataset(records, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate,
    )

    model.eval()
    model.to(device)

    temperature = temperature_scaler.temperature if temperature_scaler is not None else 1.0
    has_trunk = mahal_detector is not None and hasattr(model, "extract_trunk_embeddings")

    results: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            true_sfs = batch["true_superfamily"]
            seq_ids = batch["seq_id"]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                _ord_logits, sf_logits = model(input_ids, attention_mask)

            sf_logits_f32 = sf_logits.float()

            # Temperature-scaled logits for probabilities
            if temperature_scaler is not None:
                sf_logits_for_probs = temperature_scaler.calibrate(sf_logits_f32)
            else:
                sf_logits_for_probs = sf_logits_f32

            sf_probs = torch.softmax(sf_logits_for_probs, dim=-1).cpu().numpy()
            sf_conf = sf_probs.max(axis=1)
            sf_pred_idx = sf_probs.argmax(axis=1)

            # Energy score on raw logits
            energy_scores = EnergyScorer.score(sf_logits_f32.cpu(), T=temperature).numpy()

            # Hierarchical confidence
            pred_orders_batch = [sf_to_order_fn(idx2superfamily[int(i)]) for i in sf_pred_idx]
            hier_conf = np.array([
                sf_probs[b, order_sf_indices[pred_orders_batch[b]]].sum()
                for b in range(len(sf_pred_idx))
            ])

            # Mahalanobis scores
            if has_trunk:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    trunk_emb = model.extract_trunk_embeddings(input_ids, attention_mask)
                trunk_np = trunk_emb.float().cpu().numpy()
                mahal_scores = mahal_detector.score_batch(trunk_np)
            else:
                mahal_scores = np.full(len(seq_ids), np.nan)

            for i, seq_id in enumerate(seq_ids):
                pred_sf = idx2superfamily[int(sf_pred_idx[i])]
                true_sf = true_sfs[i]
                correct = pred_sf == true_sf
                results.append({
                    "seq_id": seq_id,
                    "true_superfamily": true_sf,
                    "pred_superfamily": pred_sf,
                    "correct": correct,
                    "sf_conf": float(sf_conf[i]),
                    "hier_conf": float(hier_conf[i]),
                    "energy_score": float(energy_scores[i]),
                    "mahal_score": float(mahal_scores[i]),
                })

    return results


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_ood(
    model_path: str,
    config: dict,
    output_dir: str,
    temperature_scaler_path: Optional[str] = None,
    mahal_detector_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    use_hierarchical_confidence: bool = True,
) -> dict:
    """Run OOD evaluation on panteon_random_test and repbase_random_test.

    Treats correctly classified sequences as in-distribution (ID) and
    misclassified sequences as pseudo-OOD.  Computes AUROC, AUPR, FPR@TPR95,
    and threshold at FPR=5% for four scorers:
      - ``sf_conf`` (negated — lower conf = more OOD)
      - ``hier_conf`` (negated)
      - ``energy_score`` (higher = more OOD)
      - ``mahal_score`` (higher = more OOD)

    Args:
        model_path: Path to ``best_model.pt`` checkpoint.
        config: Full GLARE config dict (as loaded from YAML).
        output_dir: Directory to write plots, CSV, and JSON.
        temperature_scaler_path: Optional path to ``temperature_scaler.json``.
        mahal_detector_path: Optional path to ``mahal_ood.npz``.
        device: PyTorch device string.
        batch_size: Sequences per inference batch.
        use_hierarchical_confidence: Passed through (not used for OOD scoring
            itself, but recorded in output).

    Returns:
        Dict with keys ``'panteon_random_test'`` and ``'repbase_random_test'``,
        each containing per-scorer metric dicts.
    """
    import pandas as pd
    from .model import build_model
    from .data import (
        get_tokenizer,
        build_combined_repbase_panteon_records,
        resolve_superfamily_label_subset,
    )
    from .constants import get_label_space
    from .ood import TemperatureScaler, MahalanobisOOD

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info("OOD evaluation on device: %s", _device)

    # ── Build and load model ─────────────────────────────────────────────────
    logger.info("Building model from config…")
    model = build_model(config)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing[:5])
    model.eval()
    model.to(_device)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(config["model"]["base_model"])
    max_length = config["model"]["max_length"]

    # ── OOD detectors ────────────────────────────────────────────────────────
    ts: Optional[TemperatureScaler] = None
    if temperature_scaler_path:
        ts = TemperatureScaler.from_file(temperature_scaler_path)
        logger.info("Loaded temperature scaler (T=%.4f)", ts.temperature)

    md: Optional[MahalanobisOOD] = None
    if mahal_detector_path:
        md = MahalanobisOOD.from_file(mahal_detector_path)
        logger.info("Loaded Mahalanobis detector (%d classes)", len(md.class_ids_))

    # ── Reconstruct test sets ─────────────────────────────────────────────────
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    val_fraction = float(training_cfg.get("val_fraction", 0.1))
    seed = int(training_cfg.get("seed", 7))
    label_space_name = str(data_cfg.get("label_space", "panteon30"))
    label_space = get_label_space(label_space_name)
    superfamily2idx = label_space["superfamily2idx"]
    _ls_idx2sf = label_space["idx2superfamily"]
    _ls_sf_to_order_dict = label_space["sf_to_order"]
    _ls_sf_to_order_fn = lambda sf: _ls_sf_to_order_dict[sf]
    # Build order→SF-indices mapping for the active label space
    _ls_order_sf_indices: dict[str, list[int]] = {}
    for _idx, _sf in label_space["idx2superfamily"].items():
        _ord = _ls_sf_to_order_dict[_sf]
        if _ord not in _ls_order_sf_indices:
            _ls_order_sf_indices[_ord] = []
        _ls_order_sf_indices[_ord].append(_idx)
    labels, _ = resolve_superfamily_label_subset(data_cfg)

    logger.info("Reconstructing test sets (seed=%d)…", seed)
    _train_recs, _val_recs, test_sets, _diagnostics = build_combined_repbase_panteon_records(
        data_config=data_cfg,
        val_fraction=val_fraction,
        seed=seed,
        superfamily2idx=superfamily2idx,
        label_subset_labels=labels,
        log=True,
    )

    test_set_names = ["panteon_random_test", "repbase_random_test"]
    all_results: dict[str, dict] = {}
    all_thresholds: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for ts_name in test_set_names:
        if ts_name not in test_sets:
            logger.warning("Test set '%s' not found in test_sets (keys: %s)", ts_name, list(test_sets.keys()))
            continue

        records = test_sets[ts_name]
        logger.info("Running inference on %s (%d records)…", ts_name, len(records))

        seq_results = _run_inference_on_records(
            records=records,
            model=model,
            tokenizer=tokenizer,
            device=_device,
            batch_size=batch_size,
            max_length=max_length,
            temperature_scaler=ts,
            mahal_detector=md,
            use_hierarchical_confidence=use_hierarchical_confidence,
            idx2superfamily=_ls_idx2sf,
            sf_to_order_fn=_ls_sf_to_order_fn,
            order_sf_indices=_ls_order_sf_indices,
        )

        # Convert to arrays
        correct_arr = np.array([r["correct"] for r in seq_results], dtype=bool)
        sf_conf_arr = np.array([r["sf_conf"] for r in seq_results])
        hier_conf_arr = np.array([r["hier_conf"] for r in seq_results])
        energy_arr = np.array([r["energy_score"] for r in seq_results])
        mahal_arr = np.array([r["mahal_score"] for r in seq_results])

        # y_true: 1 = OOD (misclassified), 0 = ID (correct)
        y_true = (~correct_arr).astype(int)

        n_correct = int(correct_arr.sum())
        n_incorrect = int((~correct_arr).sum())
        logger.info(
            "%s: %d correct (ID), %d incorrect (OOD), accuracy=%.3f",
            ts_name, n_correct, n_incorrect,
            n_correct / max(len(correct_arr), 1),
        )

        # Define scorers: (name, scores array, higher=more OOD)
        scorers = [
            ("sf_conf", -sf_conf_arr),       # negate: lower conf = more OOD
            ("hier_conf", -hier_conf_arr),    # negate
            ("energy_score", energy_arr),     # higher = more OOD
            ("mahal_score", mahal_arr),       # higher = more OOD
        ]

        scorer_metrics: dict[str, dict] = {}
        thresholds_for_set: dict[str, dict] = {}

        for scorer_name, scores in scorers:
            # Skip mahal if not available
            if scorer_name == "mahal_score" and np.all(np.isnan(scores)):
                logger.info("Skipping mahal_score (no detector loaded).")
                scorer_metrics[scorer_name] = {
                    "auroc": float("nan"),
                    "aupr": float("nan"),
                    "fpr_at_tpr95": float("nan"),
                    "threshold_at_fpr5": float("nan"),
                    "fpr_at_threshold": float("nan"),
                    "tpr_at_threshold": float("nan"),
                    "_y_true": y_true,
                    "_scores": scores,
                }
                continue

            metrics = _compute_ood_metrics(y_true, scores, scorer_name)
            metrics["_y_true"] = y_true
            metrics["_scores"] = scores
            scorer_metrics[scorer_name] = metrics

            # Convert negated threshold back to original scale for conf scorers
            raw_tau = metrics["threshold_at_fpr5"]
            if scorer_name in ("sf_conf", "hier_conf"):
                display_tau = -raw_tau  # back to original confidence scale
            else:
                display_tau = raw_tau

            thresholds_for_set[scorer_name] = {
                "threshold": display_tau,
                "fpr_at_threshold": metrics["fpr_at_threshold"],
                "tpr_at_threshold": metrics["tpr_at_threshold"],
            }

            logger.info(
                "%s | %s: AUROC=%.4f AUPR=%.4f FPR@TPR95=%.4f τ@FPR5%%=%.4f",
                ts_name, scorer_name,
                metrics["auroc"], metrics["aupr"],
                metrics["fpr_at_tpr95"], display_tau,
            )

            summary_rows.append({
                "test_set": ts_name,
                "scorer": scorer_name,
                "auroc": metrics["auroc"],
                "aupr": metrics["aupr"],
                "fpr_at_tpr95": metrics["fpr_at_tpr95"],
                "threshold_at_fpr5": display_tau,
            })

        all_results[ts_name] = scorer_metrics
        all_thresholds[ts_name] = thresholds_for_set

        # ── Plots ────────────────────────────────────────────────────────────
        _plot_roc_curves(scorer_metrics, ts_name, output_path)
        _plot_score_distributions(scorer_metrics, ts_name, output_path)

        # Calibration plot for sf_conf (original, not negated)
        _plot_calibration(
            sf_conf_arr, correct_arr, ts_name, output_path, label="sf_conf"
        )

        # Save per-test-set predictions CSV
        pred_df = pd.DataFrame(seq_results)
        pred_csv = output_path / f"predictions_{ts_name}.csv"
        pred_df.to_csv(str(pred_csv), index=False)
        logger.info("Predictions saved to %s", pred_csv)

    # ── Save summary CSV ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_path / "ood_eval_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    logger.info("OOD eval summary saved to %s", summary_csv)

    # ── Save thresholds JSON ─────────────────────────────────────────────────
    thresholds_json = output_path / "ood_thresholds.json"
    with open(str(thresholds_json), "w") as fh:
        json.dump(all_thresholds, fh, indent=2)
    logger.info("OOD thresholds saved to %s", thresholds_json)

    # Strip internal arrays before returning
    clean_results: dict[str, dict] = {}
    for ts_name, scorer_metrics in all_results.items():
        clean_results[ts_name] = {}
        for scorer_name, metrics in scorer_metrics.items():
            clean_results[ts_name][scorer_name] = {
                k: v for k, v in metrics.items() if not k.startswith("_")
            }

    return clean_results


# ---------------------------------------------------------------------------
# True OOD vs FASTA evaluation
# ---------------------------------------------------------------------------

def evaluate_ood_vs_fasta(
    model_path: str,
    config: dict,
    id_fasta_paths: list,
    ood_fasta_paths: list,
    output_dir: str,
    temperature_scaler_path: Optional[str] = None,
    mahal_detector_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    max_sequences_per_file: Optional[int] = None,
    use_hierarchical_confidence: bool = True,
    id_label: str = "TE",
    ood_label: str = "non-TE",
) -> dict:
    """Evaluate OOD detection against real non-TE FASTA sequences.

    Runs inference on both in-distribution (ID) TE FASTA files and
    out-of-distribution (OOD) non-TE FASTA files, then computes AUROC,
    AUPR, FPR@TPR95, and threshold at FPR=5% for four scorers.

    Args:
        model_path: Path to ``best_model.pt`` checkpoint.
        config: Full GLARE config dict (as loaded from YAML).
        id_fasta_paths: List of FASTA file paths for in-distribution (TE) sequences.
        ood_fasta_paths: List of FASTA file paths for OOD (non-TE) sequences.
        output_dir: Directory to write plots, CSV, and JSON.
        temperature_scaler_path: Optional path to ``temperature_scaler.json``.
        mahal_detector_path: Optional path to ``mahal_ood.npz``.
        device: PyTorch device string.
        batch_size: Sequences per inference batch.
        max_sequences_per_file: If set, randomly subsample each file to this
            many sequences before inference (seed=42 for reproducibility).
        use_hierarchical_confidence: Whether to compute hierarchical confidence.
        id_label: Human-readable label for the ID set (used in plots).
        ood_label: Human-readable label for the OOD set (used in plots).

    Returns:
        Dict with per-scorer metric dicts (keys: ``sf_conf``, ``hier_conf``,
        ``energy_score``, ``mahal_score``).
    """
    import pandas as pd
    from .model import build_model
    from .data import get_tokenizer
    from .ood import TemperatureScaler, MahalanobisOOD
    from .predict import predict_fasta
    from .constants import get_label_space as _get_label_space

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info("OOD vs FASTA evaluation on device: %s", _device)

    # ── Resolve label space from config ──────────────────────────────────────
    _data_cfg = config.get("data", {})
    _label_space_name: str = str(_data_cfg.get("label_space", "panteon30"))

    # ── Build and load model ─────────────────────────────────────────────────
    logger.info("Building model from config…")
    model = build_model(config)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing[:5])
    model.eval()
    model.to(_device)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(config["model"]["base_model"])
    max_length = config["model"]["max_length"]

    # ── OOD detectors ────────────────────────────────────────────────────────
    ts: Optional[TemperatureScaler] = None
    if temperature_scaler_path:
        ts = TemperatureScaler.from_file(temperature_scaler_path)
        logger.info("Loaded temperature scaler (T=%.4f)", ts.temperature)

    md: Optional[MahalanobisOOD] = None
    if mahal_detector_path:
        md = MahalanobisOOD.from_file(mahal_detector_path)
        logger.info("Loaded Mahalanobis detector (%d classes)", len(md.class_ids_))

    # ── Helper: run predict_fasta on one file (with optional subsampling) ────
    def _infer_fasta(fasta_path_str: str, label: str) -> pd.DataFrame:
        fasta_path = Path(fasta_path_str)
        tmp_path: Optional[Path] = None

        if max_sequences_per_file is not None:
            result_path = _subsample_fasta(fasta_path_str, max_sequences_per_file, seed=42)
            # _subsample_fasta returns original path if no subsampling needed
            if result_path != fasta_path:
                tmp_path = result_path
            fasta_path = result_path

        logger.info("Running inference on %s (%s)…", fasta_path, label)
        try:
            df = predict_fasta(
                model=model,
                fasta_path=fasta_path,
                tokenizer=tokenizer,
                device=_device,
                threshold=0.7,
                batch_size=batch_size,
                max_length=max_length,
                use_hierarchical_confidence=use_hierarchical_confidence,
                temperature_scaler=ts,
                mahal_detector=md,
                label_space=_label_space_name,
                sliding_window_overlap=_data_cfg.get("sliding_window_overlap", 0.5),
            )
        finally:
            # Clean up temp file if we created one
            if tmp_path is not None and tmp_path.exists():
                try:
                    os.unlink(str(tmp_path))
                except OSError:
                    pass

        logger.info("  → %d sequences inferred from %s", len(df), fasta_path_str)
        return df

    # ── Run inference on all ID and OOD files ────────────────────────────────
    id_dfs: list[pd.DataFrame] = []
    for fp in id_fasta_paths:
        df = _infer_fasta(fp, id_label)
        id_dfs.append(df)

    ood_dfs: list[pd.DataFrame] = []
    for fp in ood_fasta_paths:
        df = _infer_fasta(fp, ood_label)
        ood_dfs.append(df)

    if not id_dfs:
        raise ValueError("No ID sequences were loaded — check --id-fasta paths.")
    if not ood_dfs:
        raise ValueError("No OOD sequences were loaded — check --ood-fasta paths.")

    id_df = pd.concat(id_dfs, ignore_index=True)
    ood_df = pd.concat(ood_dfs, ignore_index=True)

    logger.info(
        "Total: %d ID sequences (%s), %d OOD sequences (%s).",
        len(id_df), id_label, len(ood_df), ood_label,
    )

    # ── Save raw prediction DataFrames ───────────────────────────────────────
    id_pred_csv = output_path / "ood_vs_fasta_predictions_id.csv"
    ood_pred_csv = output_path / "ood_vs_fasta_predictions_ood.csv"
    id_df.to_csv(str(id_pred_csv), index=False)
    ood_df.to_csv(str(ood_pred_csv), index=False)
    logger.info("ID predictions saved to %s", id_pred_csv)
    logger.info("OOD predictions saved to %s", ood_pred_csv)

    # ── Build arrays and binary labels ───────────────────────────────────────
    # y=0 → ID (TE), y=1 → OOD (non-TE)
    id_sf_conf = id_df["sf_conf"].values
    id_hier_conf = id_df["hier_conf"].values
    id_energy = id_df["energy_score"].values
    id_mahal = id_df["mahal_score"].values

    ood_sf_conf = ood_df["sf_conf"].values
    ood_hier_conf = ood_df["hier_conf"].values
    ood_energy = ood_df["energy_score"].values
    ood_mahal = ood_df["mahal_score"].values

    y_true = np.concatenate([
        np.zeros(len(id_df), dtype=int),
        np.ones(len(ood_df), dtype=int),
    ])

    sf_conf_all = np.concatenate([id_sf_conf, ood_sf_conf])
    hier_conf_all = np.concatenate([id_hier_conf, ood_hier_conf])
    energy_all = np.concatenate([id_energy, ood_energy])
    mahal_all = np.concatenate([id_mahal, ood_mahal])

    # ── Compute metrics for each scorer ──────────────────────────────────────
    # Sign convention: higher score = more OOD
    # sf_conf and hier_conf are negated (higher conf = more ID)
    scorers = [
        ("sf_conf", -sf_conf_all),
        ("hier_conf", -hier_conf_all),
        ("energy_score", energy_all),
        ("mahal_score", mahal_all),
    ]

    scorer_metrics: dict[str, dict] = {}
    thresholds_dict: dict[str, dict] = {}
    summary_rows: list[dict] = []

    id_source = ", ".join(id_fasta_paths)
    ood_source = ", ".join(ood_fasta_paths)

    for scorer_name, scores in scorers:
        # Skip mahal if not available
        if scorer_name == "mahal_score" and np.all(np.isnan(scores)):
            logger.info("Skipping mahal_score (no detector loaded).")
            scorer_metrics[scorer_name] = {
                "auroc": float("nan"),
                "aupr": float("nan"),
                "fpr_at_tpr95": float("nan"),
                "threshold_at_fpr5": float("nan"),
                "fpr_at_threshold": float("nan"),
                "tpr_at_threshold": float("nan"),
                "_y_true": y_true,
                "_scores": scores,
            }
            continue

        metrics = _compute_ood_metrics(y_true, scores, scorer_name)
        metrics["_y_true"] = y_true
        metrics["_scores"] = scores
        scorer_metrics[scorer_name] = metrics

        # Convert negated threshold back to original scale for conf scorers
        raw_tau = metrics["threshold_at_fpr5"]
        if scorer_name in ("sf_conf", "hier_conf"):
            display_tau = -raw_tau
        else:
            display_tau = raw_tau

        thresholds_dict[scorer_name] = {
            "threshold": display_tau,
            "fpr_at_threshold": metrics["fpr_at_threshold"],
            "tpr_at_threshold": metrics["tpr_at_threshold"],
        }

        logger.info(
            "%s | %s: AUROC=%.4f AUPR=%.4f FPR@TPR95=%.4f τ@FPR5%%=%.4f",
            "ood_vs_fasta", scorer_name,
            metrics["auroc"], metrics["aupr"],
            metrics["fpr_at_tpr95"], display_tau,
        )

        summary_rows.append({
            "id_source": id_source,
            "ood_source": ood_source,
            "scorer": scorer_name,
            "auroc": metrics["auroc"],
            "aupr": metrics["aupr"],
            "fpr_at_tpr95": metrics["fpr_at_tpr95"],
            "threshold_at_fpr5": display_tau,
        })

    # ── Save summary CSV ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_path / "ood_vs_fasta_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    logger.info("OOD vs FASTA summary saved to %s", summary_csv)

    # ── Save thresholds JSON ─────────────────────────────────────────────────
    thresholds_json = output_path / "ood_vs_fasta_thresholds.json"
    with open(str(thresholds_json), "w") as fh:
        json.dump(thresholds_dict, fh, indent=2)
    logger.info("OOD vs FASTA thresholds saved to %s", thresholds_json)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_roc_curves_fasta(scorer_metrics, id_label, ood_label, output_path)
    _plot_score_distributions_fasta(scorer_metrics, id_label, ood_label, output_path)

    # Strip internal arrays before returning
    clean_results: dict[str, dict] = {}
    for scorer_name, metrics in scorer_metrics.items():
        clean_results[scorer_name] = {
            k: v for k, v in metrics.items() if not k.startswith("_")
        }

    return clean_results


def evaluate_ood_testsets_vs_fasta(
    model_path: str,
    config: dict,
    ood_fasta_paths: list[str],
    output_dir: str,
    temperature_scaler_path: Optional[str] = None,
    mahal_detector_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
    max_ood_sequences_per_file: Optional[int] = 10000,
    max_id_sequences_per_testset: Optional[int] = None,
    use_hierarchical_confidence: bool = True,
    fpr_targets: Optional[list[float]] = None,
) -> dict:
    """Evaluate true OOD detection using held-out ID test sets vs OOD FASTAs.

    Reconstructs the held-out ``panteon_random_test`` and
    ``repbase_random_test`` records from the training config, treats all of
    those records as ID (label 0), and compares them against non-TE FASTA
    sequences treated as OOD (label 1).  Unlike the legacy internal evaluation,
    this does not label misclassified test-set records as pseudo-OOD.
    """
    import pandas as pd
    from .model import build_model
    from .data import (
        get_tokenizer,
        build_combined_repbase_panteon_records,
        resolve_superfamily_label_subset,
    )
    from .ood import TemperatureScaler, MahalanobisOOD
    from .predict import predict_fasta
    from .constants import get_label_space

    if not ood_fasta_paths:
        raise ValueError("At least one OOD FASTA path is required.")
    if fpr_targets is None:
        fpr_targets = _default_fpr_targets()
    fpr_targets = [float(x) for x in fpr_targets]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info("OOD testsets-vs-FASTA evaluation on device: %s", _device)

    # ── Build and load model ─────────────────────────────────────────────────
    logger.info("Building model from config…")
    model = build_model(config)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing[:5])
    model.eval()
    model.to(_device)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(config["model"]["base_model"])
    max_length = config["model"]["max_length"]

    # ── OOD detectors ────────────────────────────────────────────────────────
    ts: Optional[TemperatureScaler] = None
    if temperature_scaler_path:
        ts = TemperatureScaler.from_file(temperature_scaler_path)
        logger.info("Loaded temperature scaler (T=%.4f)", ts.temperature)

    md: Optional[MahalanobisOOD] = None
    if mahal_detector_path:
        md = MahalanobisOOD.from_file(mahal_detector_path)
        logger.info("Loaded Mahalanobis detector (%d classes)", len(md.class_ids_))

    # ── Reconstruct held-out ID test sets exactly like evaluate_ood ──────────
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    val_fraction = float(training_cfg.get("val_fraction", 0.1))
    seed = int(training_cfg.get("seed", 7))
    label_space_name = str(data_cfg.get("label_space", "panteon30"))
    label_space = get_label_space(label_space_name)
    superfamily2idx = label_space["superfamily2idx"]
    _ls_idx2sf = label_space["idx2superfamily"]
    _ls_sf_to_order_dict = label_space["sf_to_order"]
    _ls_sf_to_order_fn = lambda sf: _ls_sf_to_order_dict[sf]
    _ls_order_sf_indices: dict[str, list[int]] = {}
    for _idx, _sf in label_space["idx2superfamily"].items():
        _ord = _ls_sf_to_order_dict[_sf]
        if _ord not in _ls_order_sf_indices:
            _ls_order_sf_indices[_ord] = []
        _ls_order_sf_indices[_ord].append(_idx)
    labels, _ = resolve_superfamily_label_subset(data_cfg)

    logger.info("Reconstructing held-out test sets (seed=%d)…", seed)
    _train_recs, _val_recs, test_sets, _diagnostics = build_combined_repbase_panteon_records(
        data_config=data_cfg,
        val_fraction=val_fraction,
        seed=seed,
        superfamily2idx=superfamily2idx,
        label_subset_labels=labels,
        log=True,
    )

    # ── Inference on ID test-set records ─────────────────────────────────────
    id_result_rows: list[dict] = []
    rng = random.Random(seed)
    test_set_names = ["panteon_random_test", "repbase_random_test"]
    for ts_name in test_set_names:
        if ts_name not in test_sets:
            logger.warning("Test set '%s' not found in test_sets (keys: %s)", ts_name, list(test_sets.keys()))
            continue
        records = list(test_sets[ts_name])
        if max_id_sequences_per_testset is not None and len(records) > max_id_sequences_per_testset:
            logger.info(
                "Subsampling %d / %d ID records from %s (seed=%d).",
                max_id_sequences_per_testset, len(records), ts_name, seed,
            )
            records = rng.sample(records, max_id_sequences_per_testset)

        logger.info("Running ID inference on %s (%d records)…", ts_name, len(records))
        seq_results = _run_inference_on_records(
            records=records,
            model=model,
            tokenizer=tokenizer,
            device=_device,
            batch_size=batch_size,
            max_length=max_length,
            temperature_scaler=ts,
            mahal_detector=md,
            use_hierarchical_confidence=use_hierarchical_confidence,
            idx2superfamily=_ls_idx2sf,
            sf_to_order_fn=_ls_sf_to_order_fn,
            order_sf_indices=_ls_order_sf_indices,
        )
        for row in seq_results:
            row["source_test_set"] = ts_name
            row["ood_label"] = 0
        id_result_rows.extend(seq_results)

    if not id_result_rows:
        raise ValueError("No ID test-set records were loaded from panteon_random_test or repbase_random_test.")
    id_df = pd.DataFrame(id_result_rows)

    # ── Inference on OOD FASTA records ───────────────────────────────────────
    def _infer_ood_fasta(fasta_path_str: str) -> pd.DataFrame:
        fasta_path = Path(fasta_path_str)
        tmp_path: Optional[Path] = None
        if max_ood_sequences_per_file is not None:
            result_path = _subsample_fasta(fasta_path_str, max_ood_sequences_per_file, seed=42)
            if result_path != fasta_path:
                tmp_path = result_path
            fasta_path = result_path

        logger.info("Running OOD inference on %s…", fasta_path)
        try:
            df = predict_fasta(
                model=model,
                fasta_path=fasta_path,
                tokenizer=tokenizer,
                device=_device,
                threshold=0.7,
                batch_size=batch_size,
                max_length=max_length,
                use_hierarchical_confidence=use_hierarchical_confidence,
                temperature_scaler=ts,
                mahal_detector=md,
                label_space=label_space_name,
                sliding_window_overlap=config.get("data", {}).get("sliding_window_overlap", 0.5),
            )
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    os.unlink(str(tmp_path))
                except OSError:
                    pass
        df["ood_source"] = Path(fasta_path_str).name
        df["ood_label"] = 1
        logger.info("  → %d OOD sequences inferred from %s", len(df), fasta_path_str)
        return df

    ood_dfs = [_infer_ood_fasta(fp) for fp in ood_fasta_paths]
    if not ood_dfs:
        raise ValueError("No OOD sequences were loaded — check --ood-fasta paths.")
    ood_df = pd.concat(ood_dfs, ignore_index=True)
    if len(ood_df) == 0:
        raise ValueError("OOD FASTA inference produced zero rows — check --ood-fasta paths.")

    logger.info(
        "Total: %d ID test-set sequences, %d OOD FASTA sequences.",
        len(id_df), len(ood_df),
    )

    # ── Save prediction DataFrames ───────────────────────────────────────────
    id_pred_csv = output_path / "ood_testsets_vs_fasta_predictions_id.csv"
    ood_pred_csv = output_path / "ood_testsets_vs_fasta_predictions_ood.csv"
    id_df.to_csv(str(id_pred_csv), index=False)
    ood_df.to_csv(str(ood_pred_csv), index=False)
    logger.info("ID predictions saved to %s", id_pred_csv)
    logger.info("OOD predictions saved to %s", ood_pred_csv)

    # ── Build arrays and binary labels ───────────────────────────────────────
    y_true = np.concatenate([
        np.zeros(len(id_df), dtype=int),
        np.ones(len(ood_df), dtype=int),
    ])
    sf_conf_all = np.concatenate([id_df["sf_conf"].values, ood_df["sf_conf"].values])
    hier_conf_all = np.concatenate([id_df["hier_conf"].values, ood_df["hier_conf"].values])
    energy_all = np.concatenate([id_df["energy_score"].values, ood_df["energy_score"].values])
    mahal_all = np.concatenate([id_df["mahal_score"].values, ood_df["mahal_score"].values])

    scorers = [
        ("sf_conf", -sf_conf_all),
        ("hier_conf", -hier_conf_all),
        ("energy_score", energy_all),
        ("mahal_score", mahal_all),
    ]

    scorer_metrics: dict[str, dict] = {}
    thresholds_dict: dict[str, dict] = {}
    summary_rows: list[dict] = []
    id_counts = id_df["source_test_set"].value_counts().to_dict()
    ood_counts = ood_df["ood_source"].value_counts().to_dict()

    for scorer_name, scores in scorers:
        if scorer_name == "mahal_score" and np.all(np.isnan(scores)):
            logger.info("Skipping mahal_score (no detector loaded).")
            metrics = _nan_ood_metrics(fpr_targets)
        else:
            metrics = _compute_ood_metrics(y_true, scores, scorer_name, fpr_targets=fpr_targets)
        metrics["_y_true"] = y_true
        metrics["_scores"] = scores
        scorer_metrics[scorer_name] = metrics

        display_thresholds = _display_threshold_metrics_for_scorer(scorer_name, metrics, fpr_targets)
        thresholds_dict[scorer_name] = {
            "thresholds_at_fpr": {
                target_key: target_values
                for target_key, target_values in display_thresholds.items()
                if isinstance(target_values, dict)
            }
        }
        for key, value in display_thresholds.items():
            if not isinstance(value, dict):
                thresholds_dict[scorer_name][key] = value

        first_target = float(fpr_targets[0])
        first_key = _format_fpr_target_key(first_target)
        logger.info(
            "ood_testsets_vs_fasta | %s: AUROC=%.4f AUPR=%.4f FPR@TPR95=%.4f τ@FPR%s=%.4f",
            scorer_name,
            metrics["auroc"], metrics["aupr"], metrics["fpr_at_tpr95"],
            first_key, display_thresholds[f"threshold_at_fpr_{first_key}"],
        )

        row = {
            "comparison": "ood_testsets_vs_fasta",
            "id_source": "panteon_random_test, repbase_random_test",
            "ood_source": ", ".join(Path(fp).name for fp in ood_fasta_paths),
            "n_id": int(len(id_df)),
            "n_ood": int(len(ood_df)),
            "id_counts_by_test_set": json.dumps(id_counts, sort_keys=True),
            "ood_counts_by_source": json.dumps(ood_counts, sort_keys=True),
            "scorer": scorer_name,
            "auroc": metrics["auroc"],
            "aupr": metrics["aupr"],
            "fpr_at_tpr95": metrics["fpr_at_tpr95"],
        }
        for target in fpr_targets:
            suffix = _format_fpr_target_key(float(target))
            row[f"threshold_at_fpr_{suffix}"] = display_thresholds[f"threshold_at_fpr_{suffix}"]
            row[f"tpr_at_fpr_{suffix}"] = display_thresholds[f"tpr_at_threshold_at_fpr_{suffix}"]
            row[f"fpr_at_fpr_{suffix}"] = display_thresholds[f"fpr_at_threshold_at_fpr_{suffix}"]
        summary_rows.append(row)

    # ── Save summary and thresholds ──────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_path / "ood_testsets_vs_fasta_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    logger.info("OOD testsets-vs-FASTA summary saved to %s", summary_csv)

    thresholds_json = output_path / "ood_testsets_vs_fasta_thresholds.json"
    with open(str(thresholds_json), "w") as fh:
        json.dump(thresholds_dict, fh, indent=2)
    logger.info("OOD testsets-vs-FASTA thresholds saved to %s", thresholds_json)

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_roc_curves_fasta(
        scorer_metrics,
        "held-out TE test sets",
        "non-TE FASTA",
        output_path,
        output_filename="ood_testsets_vs_fasta_roc_curves.png",
    )
    _plot_score_distributions_fasta(
        scorer_metrics,
        "held-out TE test sets",
        "non-TE FASTA",
        output_path,
        output_filename="ood_testsets_vs_fasta_score_distributions.png",
    )

    clean_results: dict[str, dict] = {}
    for scorer_name, metrics in scorer_metrics.items():
        clean_results[scorer_name] = {
            k: v for k, v in metrics.items() if not k.startswith("_")
        }
    return clean_results


def _plot_roc_curves_fasta(
    results_per_scorer: dict,
    id_label: str,
    ood_label: str,
    output_dir: Path,
    output_filename: str = "ood_vs_fasta_roc_curves.png",
) -> None:
    """Plot ROC curves for all scorers (FASTA OOD evaluation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "sf_conf": "#1f77b4",
        "hier_conf": "#ff7f0e",
        "energy_score": "#2ca02c",
        "mahal_score": "#d62728",
    }
    labels_map = {
        "sf_conf": "SF Conf (negated)",
        "hier_conf": "Hier Conf (negated)",
        "energy_score": "Energy Score",
        "mahal_score": "Mahal Score",
    }

    for scorer, metrics in results_per_scorer.items():
        if np.isnan(metrics.get("auroc", float("nan"))):
            continue
        y_true = metrics["_y_true"]
        scores = metrics["_scores"]
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = metrics["auroc"]
        ax.plot(
            fpr, tpr,
            color=colors.get(scorer, None),
            label=f"{labels_map.get(scorer, scorer)} (AUROC={auroc:.3f})",
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {id_label} vs {ood_label}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    out_path = output_dir / output_filename
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", out_path)


def _plot_score_distributions_fasta(
    results_per_scorer: dict,
    id_label: str,
    ood_label: str,
    output_dir: Path,
    output_filename: str = "ood_vs_fasta_score_distributions.png",
) -> None:
    """Plot score distribution histograms (ID vs OOD) for FASTA evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer_names = list(results_per_scorer.keys())
    n = len(scorer_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels_map = {
        "sf_conf": "SF Conf (negated)",
        "hier_conf": "Hier Conf (negated)",
        "energy_score": "Energy Score",
        "mahal_score": "Mahal Score",
    }

    for ax, scorer in zip(axes, scorer_names):
        metrics = results_per_scorer[scorer]
        y_true = metrics["_y_true"]
        scores = metrics["_scores"]

        id_scores = scores[y_true == 0]
        ood_scores = scores[y_true == 1]

        bins = 50
        ax.hist(id_scores, bins=bins, alpha=0.6, color="#1f77b4", label=id_label, density=True)
        ax.hist(ood_scores, bins=bins, alpha=0.6, color="#d62728", label=ood_label, density=True)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(labels_map.get(scorer, scorer), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score Distributions — {id_label} vs {ood_label}", fontsize=13)
    out_path = output_dir / output_filename
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("Score distributions saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    """Command-line entry point for ``python -m glare.glare.eval_ood``."""
    import argparse

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="OOD evaluation pipeline for a trained GLARE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file used for training.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write evaluation outputs.",
    )
    parser.add_argument(
        "--temperature-scaler",
        default=None,
        help="Path to temperature_scaler.json.",
    )
    parser.add_argument(
        "--mahal-detector",
        default=None,
        help="Path to mahal_ood.npz.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--no-hierarchical-confidence",
        action="store_true",
        default=False,
        help="Use flat SF confidence instead of hierarchical order confidence.",
    )
    # ── Arguments for FASTA OOD evaluation ───────────────────────────────────
    parser.add_argument(
        "--eval-mode",
        choices=["internal", "fasta", "testsets-fasta", "both"],
        default="internal",
        help=(
            "Evaluation mode: 'internal' runs the existing pseudo-OOD evaluation "
            "on panteon_random_test / repbase_random_test; 'fasta' runs true OOD "
            "evaluation against ID and OOD FASTA files; 'testsets-fasta' runs true "
            "OOD evaluation using held-out test sets as ID against OOD FASTA files; "
            "'both' runs internal and fasta."
        ),
    )
    parser.add_argument(
        "--id-fasta",
        action="append",
        dest="id_fasta",
        default=None,
        metavar="PATH",
        help=(
            "In-distribution FASTA file (TE sequences). Can be specified multiple "
            "times. Required when --eval-mode is 'fasta' or 'both'."
        ),
    )
    parser.add_argument(
        "--ood-fasta",
        action="append",
        dest="ood_fasta",
        default=None,
        metavar="PATH",
        help=(
            "Out-of-distribution FASTA file (non-TE sequences). Can be specified "
            "multiple times. Required when --eval-mode is 'fasta' or 'both'."
        ),
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Maximum number of sequences to use per FASTA file (random subsample "
            "with seed=42). Default: use all sequences."
        ),
    )
    parser.add_argument(
        "--max-ood-sequences-per-file",
        type=int,
        default=10000,
        metavar="INT",
        help=(
            "Maximum number of OOD sequences to use per FASTA file for "
            "--eval-mode testsets-fasta (random subsample with seed=42)."
        ),
    )
    parser.add_argument(
        "--max-id-sequences-per-testset",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Maximum number of held-out ID sequences to use from each test set "
            "for --eval-mode testsets-fasta. Default: use all."
        ),
    )
    parser.add_argument(
        "--fpr-target",
        action="append",
        dest="fpr_targets",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "FPR target for testsets-fasta threshold reporting. Can be specified "
            "multiple times. Default: 0.05, 0.01, 0.005, 0.001."
        ),
    )
    parser.add_argument(
        "--id-label",
        default="TE",
        metavar="STR",
        help="Human-readable label for the ID set in plots.",
    )
    parser.add_argument(
        "--ood-label",
        default="non-TE",
        metavar="STR",
        help="Human-readable label for the OOD set in plots.",
    )

    args = parser.parse_args()

    # Validate FASTA arguments when needed
    if args.eval_mode in ("fasta", "both"):
        if not args.id_fasta:
            parser.error("--id-fasta is required when --eval-mode is 'fasta' or 'both'.")
        if not args.ood_fasta:
            parser.error("--ood-fasta is required when --eval-mode is 'fasta' or 'both'.")
    if args.eval_mode == "testsets-fasta" and not args.ood_fasta:
        parser.error("--ood-fasta is required when --eval-mode is 'testsets-fasta'.")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    use_hier = not args.no_hierarchical_confidence
    fpr_targets = args.fpr_targets if args.fpr_targets is not None else _default_fpr_targets()

    # ── Internal pseudo-OOD evaluation ───────────────────────────────────────
    if args.eval_mode in ("internal", "both"):
        results = evaluate_ood(
            model_path=args.model_path,
            config=cfg,
            output_dir=args.output_dir,
            temperature_scaler_path=args.temperature_scaler,
            mahal_detector_path=args.mahal_detector,
            device=args.device,
            batch_size=args.batch_size,
            use_hierarchical_confidence=use_hier,
        )

        print("\nOOD Evaluation Complete (internal pseudo-OOD):")
        for ts_name, scorer_metrics in results.items():
            print(f"\n  {ts_name}:")
            for scorer_name, metrics in scorer_metrics.items():
                auroc = metrics.get("auroc", float("nan"))
                aupr = metrics.get("aupr", float("nan"))
                fpr95 = metrics.get("fpr_at_tpr95", float("nan"))
                tau = metrics.get("threshold_at_fpr5", float("nan"))
                print(
                    f"    {scorer_name:15s}: AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                    f"FPR@TPR95={fpr95:.4f}  τ@FPR5%={tau:.4f}"
                )
        print(f"\nOutputs written to: {args.output_dir}")

    # ── True OOD: held-out test sets as ID vs OOD FASTA ──────────────────────
    if args.eval_mode == "testsets-fasta":
        testsets_fasta_results = evaluate_ood_testsets_vs_fasta(
            model_path=args.model_path,
            config=cfg,
            ood_fasta_paths=args.ood_fasta,
            output_dir=args.output_dir,
            temperature_scaler_path=args.temperature_scaler,
            mahal_detector_path=args.mahal_detector,
            device=args.device,
            batch_size=args.batch_size,
            max_ood_sequences_per_file=args.max_ood_sequences_per_file,
            max_id_sequences_per_testset=args.max_id_sequences_per_testset,
            use_hierarchical_confidence=use_hier,
            fpr_targets=fpr_targets,
        )

        print("\nOOD Testsets vs FASTA Evaluation Complete:")
        first_target = float(fpr_targets[0])
        first_suffix = _format_fpr_target_key(first_target)
        for scorer_name, metrics in testsets_fasta_results.items():
            auroc = metrics.get("auroc", float("nan"))
            aupr = metrics.get("aupr", float("nan"))
            fpr95 = metrics.get("fpr_at_tpr95", float("nan"))
            tau = metrics.get(f"threshold_at_fpr_{first_suffix}", metrics.get("threshold_at_fpr5", float("nan")))
            if scorer_name in ("sf_conf", "hier_conf"):
                tau = -tau
            print(
                f"  {scorer_name:15s}: AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                f"FPR@TPR95={fpr95:.4f}  τ@FPR{first_target:g}={tau:.4f}"
            )
        print(f"\nOutputs written to: {args.output_dir}")

    # ── True OOD vs FASTA evaluation ─────────────────────────────────────────
    if args.eval_mode in ("fasta", "both"):
        fasta_results = evaluate_ood_vs_fasta(
            model_path=args.model_path,
            config=cfg,
            id_fasta_paths=args.id_fasta,
            ood_fasta_paths=args.ood_fasta,
            output_dir=args.output_dir,
            temperature_scaler_path=args.temperature_scaler,
            mahal_detector_path=args.mahal_detector,
            device=args.device,
            batch_size=args.batch_size,
            max_sequences_per_file=args.max_sequences,
            use_hierarchical_confidence=use_hier,
            id_label=args.id_label,
            ood_label=args.ood_label,
        )

        print(f"\nOOD vs FASTA Evaluation Complete ({args.id_label} vs {args.ood_label}):")
        for scorer_name, metrics in fasta_results.items():
            auroc = metrics.get("auroc", float("nan"))
            aupr = metrics.get("aupr", float("nan"))
            fpr95 = metrics.get("fpr_at_tpr95", float("nan"))
            tau = metrics.get("threshold_at_fpr5", float("nan"))
            print(
                f"  {scorer_name:15s}: AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                f"FPR@TPR95={fpr95:.4f}  τ@FPR5%={tau:.4f}"
            )
        print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    _main()
