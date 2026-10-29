"""Standalone post-hoc OOD calibration for GLARE.

This module can be run after training to fit and save:
  - :class:`~glare.ood.TemperatureScaler` (temperature_scaler.json)
  - :class:`~glare.ood.MahalanobisOOD` (mahal_ood.npz)

It is useful when the model was trained without ``ood.fit_calibration: true``
in the config, or when you want to re-calibrate on a different validation set.

Usage (as a module)::

    python -m glare.glare.calibrate \\
        --model-path glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output/best_model.pt \\
        --config glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment.yaml \\
        --output-dir glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_output

Or import and call :func:`calibrate_model` programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------

def calibrate_model(
    model_path: str,
    config: dict,
    output_dir: str,
    device: str = "cuda",
    batch_size: Optional[int] = None,
) -> dict:
    """Fit and save TemperatureScaler and MahalanobisOOD for a trained GLARE model.

    Loads the best checkpoint, runs the validation set through the model to
    collect logits (for temperature scaling) and the training set to collect
    trunk embeddings (for Mahalanobis OOD).

    Args:
        model_path: Path to the ``best_model.pt`` checkpoint.
        config: Full GLARE config dict (as loaded from YAML).
        output_dir: Directory where ``temperature_scaler.json`` and
            ``mahal_ood.npz`` will be written.
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
        batch_size: Override ``training.batch_size`` from config.

    Returns:
        Dict with keys:
          - ``"temperature"`` (float): fitted temperature T.
          - ``"mahal_n_classes"`` (int): number of classes in the Mahalanobis
            detector (0 if trunk extraction is not supported).
    """
    from .model import build_model
    from .data import (
        get_tokenizer,
        build_combined_repbase_panteon_records,
        make_sliding_window_dataloader,
        resolve_superfamily_label_subset,
        split_panteon_train_val_records,
        get_label_space as _get_label_space_fn,
    )
    from .constants import get_label_space
    from .ood import TemperatureScaler, MahalanobisOOD

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info("Calibrating on device: %s", _device)

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
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── Data ─────────────────────────────────────────────────────────────────
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    _batch_size = batch_size or training_cfg.get("batch_size", 16)
    max_length = config["model"]["max_length"]
    val_fraction = float(training_cfg.get("val_fraction", 0.1))
    seed = int(training_cfg.get("seed", 7))
    num_workers = 2
    sliding_window_overlap = float(data_cfg.get("sliding_window_overlap", 0.5))

    label_space_name = str(data_cfg.get("label_space", "panteon30"))
    label_space = get_label_space(label_space_name)
    superfamily2idx = label_space["superfamily2idx"]
    order2idx = label_space["order2idx"]

    labels, _ = resolve_superfamily_label_subset(data_cfg)

    logger.info("Loading training/validation data…")
    train_recs, val_recs, _test_sets, _diagnostics = build_combined_repbase_panteon_records(
        data_config=data_cfg,
        val_fraction=val_fraction,
        seed=seed,
        superfamily2idx=superfamily2idx,
        label_subset_labels=labels,
        log=True,
    )

    logger.info("Building val DataLoader (%d records)…", len(val_recs))
    val_loader, _n_val = make_sliding_window_dataloader(
        val_recs, tokenizer, max_length, pad_token_id,
        batch_size=_batch_size,
        num_workers=num_workers,
        overlap=sliding_window_overlap,
        order2idx=order2idx,
        superfamily2idx=superfamily2idx,
    )

    logger.info("Building train DataLoader (%d records)…", len(train_recs))
    train_loader, _n_train = make_sliding_window_dataloader(
        train_recs, tokenizer, max_length, pad_token_id,
        batch_size=_batch_size,
        num_workers=num_workers,
        overlap=sliding_window_overlap,
        order2idx=order2idx,
        superfamily2idx=superfamily2idx,
    )

    # ── Temperature Scaling ──────────────────────────────────────────────────
    logger.info("Collecting val logits for temperature scaling…")
    all_val_logits: list[torch.Tensor] = []
    all_val_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch["input_ids"].to(_device, non_blocking=True)
            b_attn_mask = batch["attention_mask"].to(_device, non_blocking=True)
            b_sf_labels = batch["superfamily_label"]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(_device.type == "cuda"),
            ):
                _ord_logits, sf_logits = model(b_input_ids, b_attn_mask)
            all_val_logits.append(sf_logits.float().cpu())
            all_val_labels.append(b_sf_labels.cpu())

    val_logits_cat = torch.cat(all_val_logits, dim=0)
    val_labels_cat = torch.cat(all_val_labels, dim=0)

    scaler = TemperatureScaler()
    fitted_T = scaler.fit(val_logits_cat, val_labels_cat)
    scaler_path = output_path / "temperature_scaler.json"
    scaler.save(str(scaler_path))
    logger.info("Temperature scaler saved to %s (T=%.4f)", scaler_path, fitted_T)

    # ── Mahalanobis OOD ──────────────────────────────────────────────────────
    mahal_n_classes = 0

    if hasattr(model, "extract_trunk_embeddings"):
        logger.info("Collecting train trunk embeddings for Mahalanobis OOD…")
        all_train_emb: list[np.ndarray] = []
        all_train_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in train_loader:
                b_input_ids = batch["input_ids"].to(_device, non_blocking=True)
                b_attn_mask = batch["attention_mask"].to(_device, non_blocking=True)
                b_sf_labels = batch["superfamily_label"].numpy()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=(_device.type == "cuda"),
                ):
                    trunk_emb = model.extract_trunk_embeddings(b_input_ids, b_attn_mask)
                all_train_emb.append(trunk_emb.float().cpu().numpy())
                all_train_labels.append(b_sf_labels)

        train_emb_np = np.concatenate(all_train_emb, axis=0)
        train_labels_np = np.concatenate(all_train_labels, axis=0)

        mahal = MahalanobisOOD()
        mahal.fit(train_emb_np, train_labels_np)
        mahal_path = output_path / "mahal_ood.npz"
        mahal.save(str(mahal_path))
        mahal_n_classes = int(len(mahal.class_ids_))
        logger.info(
            "Mahalanobis OOD detector saved to %s (%d classes)",
            mahal_path, mahal_n_classes,
        )
    else:
        logger.warning(
            "Model does not expose extract_trunk_embeddings(); "
            "skipping MahalanobisOOD fitting."
        )

    return {
        "temperature": fitted_T,
        "mahal_n_classes": mahal_n_classes,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    """Command-line entry point for ``python -m glare.glare.calibrate``."""
    import argparse

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Post-hoc OOD calibration for a trained GLARE model.",
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
        help="Directory to write temperature_scaler.json and mahal_ood.npz.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size from config.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    results = calibrate_model(
        model_path=args.model_path,
        config=cfg,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\nCalibration complete:")
    print(f"  Temperature T = {results['temperature']:.4f}")
    print(f"  Mahalanobis classes = {results['mahal_n_classes']}")
    print(f"  Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    _main()
