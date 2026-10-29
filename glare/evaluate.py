"""Evaluation for GLARE — supports both PanTEon and legacy Repbase workflows.

PanTEon workflow (primary):
  - No confidence threshold — all sequences are classified.
  - Metric: weighted F1 on the 10% test split (seed=7).
  - Use :func:`evaluate_panteon_test` or :func:`evaluate_panteon_full`.

Legacy Repbase workflow:
  - Confidence threshold applied to hierarchical order confidence.
  - 5 holdout species groups.
  - Use :func:`evaluate_holdout_groups` (kept for backward compatibility).
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from .constants import (
    IDX2SUPERFAMILY,
    IDX2ORDER,
    SUPERFAMILY_LABELS,
    ORDER_LABELS,
    SF_TO_ORDER,
    get_label_space,
    sf_to_order,
)
from .data import (
    build_taxonomy_group_test_sets,
    filter_panteon_records_by_taxonomy,
    filter_records_by_superfamilies,
    filter_records_by_superfamily_min_count,
    format_split_stats,
    load_external_repbase_fasta_for_panteon,
    load_repbase_dir_for_label_space,
    map_records_to_configured_label_space,
    parse_ref_file,
    load_panteon_dataset,
    make_sliding_window_dataloader,
    make_test_dataloader,
    resolve_superfamily_label_subset,
    split_panteon_records,
    split_stratified_train_val_test_records,
)
from .inference import MeanLogitAccumulator

# ---------------------------------------------------------------------------
# Pre-compute order → SF indices mapping (module-level, computed once)
# ---------------------------------------------------------------------------

# Maps each order name to the list of SF indices belonging to that order.
# Used for hierarchical confidence: order_conf = sum(sf_probs[ORDER_SF_INDICES[order]])
ORDER_SF_INDICES: dict[str, list[int]] = {}
for _idx, _sf in enumerate(SUPERFAMILY_LABELS):
    _order = SF_TO_ORDER.get(_sf, "")
    if _order:
        if _order not in ORDER_SF_INDICES:
            ORDER_SF_INDICES[_order] = []
        ORDER_SF_INDICES[_order].append(_idx)


def _order_sf_indices_for_label_space(label_space: dict) -> dict[str, list[int]]:
    """Build order → superfamily-index lists for a label-space dict."""
    sf_to_order_map = label_space["sf_to_order"]
    indices: dict[str, list[int]] = {}
    for idx, sf in enumerate(label_space["superfamily_labels"]):
        order = sf_to_order_map.get(sf, "")
        if order:
            indices.setdefault(order, []).append(idx)
    return indices


def _label_space_from_config(config: dict, default: str = "panteon30") -> dict:
    """Resolve active evaluation label space from config."""
    data_cfg = config.get("data", {})
    return get_label_space(data_cfg.get("label_space") or data_cfg.get("repbase_label_space", default))


def _repbase_stratified_random_split_enabled(config: dict) -> bool:
    """Return whether external Repbase eval should reproduce a random test fold."""
    split_config = config.get("data", {}).get("repbase_split") or {}
    if isinstance(split_config, str):
        mode = split_config
    else:
        mode = split_config.get("mode", "external_holdout")
    return str(mode).strip().lower().replace("-", "_") in {
        "stratified_random",
        "stratified_random_test",
        "random_stratified",
        "random",
    }


def _repbase_eval_diagnostics_from_records(
    records: list[dict],
    source_diagnostics: dict,
    split_stats: dict,
) -> dict:
    """Build Repbase diagnostics for a reproduced stratified random test fold."""
    raw_category_counts: Counter[str] = Counter()
    repeatmasker_counts: Counter[str] = Counter()
    sf_counts: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    mapped_class_rows: list[dict] = []

    for rec in records:
        raw_category = str(rec.get("original_category", rec.get("category", "")))
        repeatmasker_label = str(rec.get("repeatmasker_label", rec.get("superfamily", "")))
        superfamily = str(rec.get("superfamily", ""))
        order = str(rec.get("order", ""))
        raw_category_counts[raw_category] += 1
        repeatmasker_counts[repeatmasker_label] += 1
        sf_counts[superfamily] += 1
        order_counts[order] += 1
        mapped_class_rows.append({
            "name": rec.get("name", ""),
            "species": rec.get("species", ""),
            "original_category": raw_category,
            "repeatmasker_label": repeatmasker_label,
            "superfamily": superfamily,
            "panteon_superfamily": superfamily,
            "order": order,
            "panteon_order": order,
            "sequence_length": len(rec.get("sequence", "")),
        })

    return {
        **source_diagnostics,
        "loaded_records": len(records),
        "mapped_records": len(records),
        "skipped_records": 0,
        "source_loaded_records": source_diagnostics.get("loaded_records"),
        "source_mapped_records": source_diagnostics.get("mapped_records"),
        "source_skipped_records": source_diagnostics.get("skipped_records"),
        "raw_category_counts": dict(sorted(raw_category_counts.items())),
        "repeatmasker_counts": dict(sorted(repeatmasker_counts.items())),
        "panteon_superfamily_counts": dict(sorted(sf_counts.items())),
        "panteon_order_counts": dict(sorted(order_counts.items())),
        "skipped_reason_counts": {},
        "skipped_category_counts": {},
        "skipped_repeatmasker_counts": {},
        "mapped_class_rows": mapped_class_rows,
        "repbase_split_mode": "stratified_random",
        "repbase_split_stats": split_stats,
    }


# ---------------------------------------------------------------------------
# PanTEon metrics
# ---------------------------------------------------------------------------

def compute_panteon_metrics(
    true_sf: list[str],
    pred_sf: list[str],
    sf_to_order_map: dict[str, str] | None = None,
) -> dict:
    """Compute weighted F1, per-class F1, and accuracy for superfamily predictions.

    Order predictions are derived from superfamily predictions via
    :func:`~glare.constants.sf_to_order`.

    Args:
        true_sf: Ground-truth superfamily tokens (e.g. ``["GYPSY", "L1", ...]``).
        pred_sf: Predicted superfamily tokens.

    Returns:
        Dict with keys:
        ``sf_f1``, ``order_f1``, ``accuracy``, ``n_total``,
        ``per_class_sf_f1`` (dict SF→F1), ``per_class_order_f1`` (dict order→F1).
    """
    sf_to_order_map = sf_to_order_map or SF_TO_ORDER
    n_total = len(true_sf)

    sf_f1 = float(
        f1_score(true_sf, pred_sf, average="weighted", zero_division=0)
    )
    accuracy = float(np.mean([t == p for t, p in zip(true_sf, pred_sf)]))

    # Derive order labels
    true_order = [sf_to_order_map.get(s, s.split("/", 1)[0] if "/" in s else s) for s in true_sf]
    pred_order = [sf_to_order_map.get(s, s.split("/", 1)[0] if "/" in s else s) for s in pred_sf]
    order_f1 = float(
        f1_score(true_order, pred_order, average="weighted", zero_division=0)
    )

    # Per-class SF F1
    present_sfs = sorted(set(true_sf))
    per_class_sf_f1_arr = f1_score(
        true_sf, pred_sf,
        labels=present_sfs,
        average=None,
        zero_division=0,
    )
    per_class_sf_f1 = {sf: float(f1) for sf, f1 in zip(present_sfs, per_class_sf_f1_arr)}

    # Per-class order F1
    present_orders = sorted(set(true_order))
    per_class_order_f1_arr = f1_score(
        true_order, pred_order,
        labels=present_orders,
        average=None,
        zero_division=0,
    )
    per_class_order_f1 = {
        o: float(f1) for o, f1 in zip(present_orders, per_class_order_f1_arr)
    }

    # Confusion matrix
    cm_labels = sorted(set(true_sf) | set(pred_sf))
    cm = confusion_matrix(true_sf, pred_sf, labels=cm_labels)

    return {
        "sf_f1": sf_f1,
        "order_f1": order_f1,
        "accuracy": accuracy,
        "n_total": n_total,
        "per_class_sf_f1": per_class_sf_f1,
        "per_class_order_f1": per_class_order_f1,
        "confusion_matrix": cm,
        "confusion_matrix_labels": cm_labels,
    }


# ---------------------------------------------------------------------------
# PanTEon test-set evaluation
# ---------------------------------------------------------------------------

def _standalone_sliding_window_enabled(config: dict) -> bool:
    """Return whether standalone PanTEon eval should use sliding windows.

    Standalone evaluation historically used the non-sliding test loader.  To
    preserve that behavior for older/non-sliding configs, enable the
    sliding-window path only when the config explicitly contains
    ``data.sliding_window_overlap``.
    """
    data_config = config.get("data", {})
    return (
        "sliding_window_overlap" in data_config
        and data_config.get("sliding_window_overlap") is not None
    )


@torch.no_grad()
def evaluate_panteon_test(
    model,
    test_loader,
    device: torch.device,
    config: dict,
    n_sequences: int | None = None,
) -> dict:
    """Run inference on the PanTEon test set and return aggregate metrics.

    No confidence threshold is applied — all sequences are classified.

    When *n_sequences* is provided the loader is assumed to come from a
    :class:`~glare.data.SlidingWindowDataset` whose batches contain a
    ``seq_idx`` field.  Logits from overlapping windows are summed per
    sequence and the argmax is taken over the averaged logits (same
    aggregation used by ``eval_epoch`` in ``train.py``).

    Args:
        model: Trained :class:`~glare.model.GLAREModel`.
        test_loader: DataLoader for the test split.
        device: Inference device.
        config: Full config dict (used for ``evaluation.use_hierarchical_confidence``).
        n_sequences: Total number of *original* sequences when using a
            sliding-window DataLoader.  Pass ``None`` (default) for the
            standard per-window case (backward compatible).

    Returns:
        Dict with keys: ``sf_f1``, ``order_f1``, ``accuracy``, ``n_total``,
        ``per_class_sf_f1``, ``per_class_order_f1``.
    """
    model.eval()
    model.to(device)
    label_space = _label_space_from_config(config, default="panteon30")
    idx2sf = label_space["idx2superfamily"]
    sf_to_order_map = label_space["sf_to_order"]

    if n_sequences is not None:
        # --- Sliding-window aggregation path ---
        accumulator = MeanLogitAccumulator(n_sequences, device="cpu")

        # We also need per-sequence ground-truth labels.
        seq_true_labels: list[int | None] = [None] * n_sequences

        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            seq_idx = batch["seq_idx"].to(device)          # (B,) long
            sf_true_idx = batch["superfamily_label"]       # (B,) cpu

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                order_logits, sf_logits = model(input_ids, attention_mask)

            accumulator.add(seq_idx, order_logits, sf_logits)

            # Store ground-truth label for each sequence (same for all windows).
            for b_i, s_i in enumerate(seq_idx.cpu().tolist()):
                seq_true_labels[s_i] = int(sf_true_idx[b_i])

        # Average raw logits and take argmax. Missing records are rejected by
        # the shared accumulator instead of being hidden by clamped counts.
        _order_logits_mean, sf_logits_mean = accumulator.means()
        sf_pred_idx = sf_logits_mean.argmax(dim=-1).tolist()

        all_sf_preds = [idx2sf[int(i)] for i in sf_pred_idx]
        all_sf_true = [idx2sf[int(i)] for i in seq_true_labels]  # type: ignore[arg-type]

    else:
        # --- Standard per-batch argmax path (backward compatible) ---
        all_sf_preds: list[str] = []
        all_sf_true: list[str] = []

        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            sf_true_idx = batch["superfamily_label"].cpu().numpy()

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                _order_logits, sf_logits = model(input_ids, attention_mask)

            sf_pred_idx = sf_logits.argmax(dim=-1).cpu().numpy()

            all_sf_preds.extend([idx2sf[int(i)] for i in sf_pred_idx])
            all_sf_true.extend([idx2sf[int(i)] for i in sf_true_idx])

    return compute_panteon_metrics(all_sf_true, all_sf_preds, sf_to_order_map=sf_to_order_map)


def _write_panteon_metrics_outputs(
    metrics: dict,
    label_space: dict,
    output_dir: Path,
    prefix: str = "panteon",
    title: str = "PanTEon Test Results",
) -> pd.DataFrame:
    """Write PanTEon metric tables and confusion-matrix outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sf, f1 in sorted(metrics["per_class_sf_f1"].items()):
        order = label_space["sf_to_order"].get(sf, "UNKNOWN")
        rows.append({
            "superfamily": sf,
            "order": order,
            "sf_f1": f1,
            "order_f1": metrics["per_class_order_f1"].get(order, float("nan")),
        })
    per_class_df = pd.DataFrame(rows)

    per_class_path = output_dir / f"{prefix}_per_class_f1.csv"
    per_class_df.to_csv(per_class_path, index=False)

    summary = {
        "sf_f1": metrics["sf_f1"],
        "order_f1": metrics["order_f1"],
        "accuracy": metrics["accuracy"],
        "n_total": metrics["n_total"],
    }
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    cm_labels = metrics["confusion_matrix_labels"]
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=cm_labels, columns=cm_labels)
    cm_path = output_dir / f"{prefix}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    cm_arr = metrics["confusion_matrix"]
    n_labels = len(cm_labels)
    cm_norm = cm_arr.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    fig, ax = plt.subplots()
    fig.set_size_inches(max(8, n_labels * 0.5), max(6, n_labels * 0.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n_labels),
        yticks=range(n_labels),
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation="vertical")
    thresh = cm_norm.max() / 2.0
    for i in range(n_labels):
        for j in range(n_labels):
            ax.text(
                j, i, str(cm_arr[i, j]),
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=max(4, min(8, 80 // n_labels)),
            )
    fig.tight_layout()
    cm_fig_path = output_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix figure → {cm_fig_path}")

    print(f"\n=== {title} ===")
    print(f"  Superfamily weighted F1 : {metrics['sf_f1']:.4f}  ({metrics['sf_f1']*100:.2f}%)")
    print(f"  Order weighted F1       : {metrics['order_f1']:.4f}  ({metrics['order_f1']*100:.2f}%)")
    print(f"  Accuracy                : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  N test sequences        : {metrics['n_total']}")
    print(f"\nPer-class F1 saved to    : {per_class_path}")
    print(f"Summary saved to         : {summary_path}")
    print(f"Confusion matrix saved to: {cm_path}")

    return per_class_df


def _apply_panteon_eval_pre_split_filters(
    records: list[dict],
    *,
    label_subset_labels: list[str] | None,
    data_config: dict,
) -> list[dict]:
    """Apply standalone PanTEon evaluation filters before reproducing the split."""
    if label_subset_labels is not None:
        records, subset_filter_stats = filter_records_by_superfamilies(
            records,
            label_subset_labels,
            dataset_name="PanTEon all records before standalone eval split",
        )
        print(
            "PanTEon label subset filter: "
            f"kept={subset_filter_stats['kept_records']}/"
            f"{subset_filter_stats['total_records']} "
            f"skipped={subset_filter_stats['skipped_records']}"
        )
        if not records:
            raise ValueError(
                "data.label_subset removed all PanTEon records before evaluation split."
            )

    taxonomy_train_filter = data_config.get("taxonomy_train_filter") or data_config.get("taxonomy_filter")
    records, taxonomy_filter_stats = filter_panteon_records_by_taxonomy(
        records,
        taxonomy_train_filter,
        filter_name="data.taxonomy_train_filter",
    )
    if taxonomy_filter_stats.get("taxonomy_filter_enabled"):
        print(
            "PanTEon taxonomy train/eval filter: "
            f"{taxonomy_filter_stats.get('filter_column')}="
            f"{taxonomy_filter_stats.get('filter_values')} "
            f"kept={taxonomy_filter_stats.get('matched_records')}/"
            f"{taxonomy_filter_stats.get('total_records')} "
            f"metadata_species={taxonomy_filter_stats.get('metadata_species')} "
            f"matched_fasta_species={taxonomy_filter_stats.get('matched_fasta_species')}"
        )
    if not records:
        raise ValueError("data.taxonomy_train_filter removed all PanTEon records before evaluation split.")

    records, min_count_stats = filter_records_by_superfamily_min_count(
        records,
        data_config.get("min_superfamily_count"),
        dataset_name="PanTEon records after standalone eval taxonomy filter",
    )
    if min_count_stats.get("enabled"):
        print(
            "PanTEon min superfamily count filter: "
            f"min_count={min_count_stats.get('min_count')} "
            f"kept_superfamilies={min_count_stats.get('kept_superfamily_count')} "
            f"kept={min_count_stats.get('kept_records')}/"
            f"{min_count_stats.get('total_records')} "
            f"skipped={min_count_stats.get('skipped_records')}"
        )
    if not records:
        raise ValueError("data.min_superfamily_count removed all PanTEon records before evaluation split.")
    return records


def evaluate_panteon_full(
    model,
    fasta_path: Path,
    tokenizer,
    device: torch.device,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """Full PanTEon evaluation pipeline.

    Loads the test split from *fasta_path*, runs inference, computes metrics,
    saves a per-class F1 CSV to *output_dir*, and returns a summary DataFrame.

    Args:
        model: Trained :class:`~glare.model.GLAREModel`.
        fasta_path: Path to the PanTEon FASTA file.
        tokenizer: NTv3 tokenizer instance.
        device: Inference device.
        config: Full config dict.
        output_dir: Directory to save results CSV.

    Returns:
        :class:`pandas.DataFrame` with per-class F1 scores (one row per SF).
    """
    batch_size = config["training"]["batch_size"]
    max_length = config["model"]["max_length"]
    seed = config["training"].get("seed", 7)
    val_fraction = config["training"].get("val_fraction", 0.1)
    num_workers = 2

    taxonomy_holdout = config.get("data", {}).get("taxonomy_holdout")
    label_space = _label_space_from_config(config, default="panteon30")
    panteon_label_mapping = config.get("data", {}).get("panteon_label_mapping")
    label_subset_labels, label_subset_stats = resolve_superfamily_label_subset(
        config.get("data", {})
    )
    if label_subset_labels is not None:
        print(
            "Superfamily label subset enabled: "
            f"mode={(label_subset_stats or {}).get('mode')} "
            f"labels={len(label_subset_labels)} [{','.join(label_subset_labels)}]"
        )
    n_sequences: int | None = None

    data_config = config.get("data", {})
    rc_eval = bool(data_config.get("rc_eval", False))
    if _standalone_sliding_window_enabled(config):
        records = load_panteon_dataset(
            fasta_path,
            label_mapping_path=Path(str(panteon_label_mapping)) if panteon_label_mapping else None,
            label_space_name=str(label_space["name"]),
        )
        records = _apply_panteon_eval_pre_split_filters(
            records,
            label_subset_labels=label_subset_labels,
            data_config=data_config,
        )
        _train_recs, _val_recs, test_records, split_stats = split_panteon_records(
            records=records,
            val_fraction=val_fraction,
            seed=seed,
            taxonomy_holdout=taxonomy_holdout,
            superfamily2idx=label_space["superfamily2idx"],
        )
        print("PanTEon test split:")
        print(format_split_stats(split_stats))

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        sliding_window_overlap = float(config["data"].get("sliding_window_overlap", 0.5))
        test_loader, n_sequences = make_sliding_window_dataloader(
            records=test_records,
            tokenizer=tokenizer,
            max_length=max_length,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
            num_workers=num_workers,
            overlap=sliding_window_overlap,
            order2idx=label_space["order2idx"],
            superfamily2idx=label_space["superfamily2idx"],
            rc_eval=rc_eval,
        )
        print(
            "PanTEon standalone evaluation: using sliding-window test evaluation "
            f"(overlap={sliding_window_overlap}, n_sequences={n_sequences}, "
            f"n_windows={len(test_loader.dataset)})"
        )
    else:
        taxonomy_train_filter = data_config.get("taxonomy_train_filter") or data_config.get("taxonomy_filter")
        if label_subset_labels is not None or taxonomy_train_filter:
            records = load_panteon_dataset(
                fasta_path,
                label_mapping_path=Path(str(panteon_label_mapping)) if panteon_label_mapping else None,
                label_space_name=str(label_space["name"]),
            )
            records = _apply_panteon_eval_pre_split_filters(
                records,
                label_subset_labels=label_subset_labels,
                data_config=data_config,
            )
            _train_recs, _val_recs, test_records, split_stats = split_panteon_records(
                records=records,
                val_fraction=val_fraction,
                seed=seed,
                taxonomy_holdout=taxonomy_holdout,
                superfamily2idx=label_space["superfamily2idx"],
            )
            print("PanTEon test split:")
            print(format_split_stats(split_stats))
            from torch.utils.data import DataLoader
            from .data import TEDataset

            test_ds = TEDataset(
                test_records,
                tokenizer,
                max_length=max_length,
                order2idx=label_space["order2idx"],
                superfamily2idx=label_space["superfamily2idx"],
                rc_eval=rc_eval,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )
        else:
            test_loader, test_records = make_test_dataloader(
                fasta_path=fasta_path,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                seed=seed,
                val_fraction=val_fraction,
                num_workers=num_workers,
                taxonomy_holdout=taxonomy_holdout,
            )
        print(
            "PanTEon standalone evaluation: using non-sliding test evaluation "
            f"(n_sequences={len(test_records)})"
        )

    metrics = evaluate_panteon_test(
        model,
        test_loader,
        device,
        config,
        n_sequences=n_sequences,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    per_class_df = _write_panteon_metrics_outputs(
        metrics=metrics,
        label_space=label_space,
        output_dir=output_dir,
        prefix="panteon",
        title="PanTEon Test Results",
    )

    taxonomy_test_groups = data_config.get("taxonomy_test_groups") or []
    if taxonomy_test_groups:
        grouped_test_sets, taxonomy_group_stats = build_taxonomy_group_test_sets(
            test_records,
            taxonomy_test_groups,
        )
        print(
            "PanTEon taxonomy test groups: "
            f"{len(grouped_test_sets)} groups from {taxonomy_group_stats['total_records']} held-out test records"
        )
        for test_name, group_records in grouped_test_sets.items():
            group_stats = taxonomy_group_stats["groups"].get(test_name, {})
            print(
                f"  [{test_name}] n={len(group_records)} "
                f"metadata_filter={group_stats.get('filter_column')}={group_stats.get('filter_values')}"
            )
            group_loader, group_n_sequences = make_sliding_window_dataloader(
                records=group_records,
                tokenizer=tokenizer,
                max_length=max_length,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
                num_workers=num_workers,
                overlap=float(data_config.get("sliding_window_overlap", 0.5)),
                order2idx=label_space["order2idx"],
                superfamily2idx=label_space["superfamily2idx"],
                rc_eval=rc_eval,
            )
            group_metrics = evaluate_panteon_test(
                model,
                group_loader,
                device,
                config,
                n_sequences=group_n_sequences,
            )
            _write_panteon_metrics_outputs(
                metrics=group_metrics,
                label_space=label_space,
                output_dir=output_dir,
                prefix=test_name,
                title=f"PanTEon Test Results [{test_name}]",
            )

    return per_class_df


@torch.no_grad()
def predict_panteon_records(
    model,
    records: list[dict],
    tokenizer,
    device: torch.device,
    config: dict,
) -> pd.DataFrame:
    """Predict PanTEon-vocabulary labels for a supplied record list.

    This helper is used by external datasets that are not split by the PanTEon
    benchmark protocol.  If ``data.sliding_window_overlap`` is present in the
    config, each full sequence is evaluated with the same sliding-window logit
    averaging path as standalone PanTEon evaluation.
    """
    from torch.utils.data import DataLoader
    from .data import TEDataset

    batch_size = config["training"]["batch_size"]
    max_length = config["model"]["max_length"]
    num_workers = 2
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    label_space = _label_space_from_config(config, default="panteon30")
    idx2sf = label_space["idx2superfamily"]
    sf_to_order_map = label_space["sf_to_order"]
    order_sf_indices = _order_sf_indices_for_label_space(label_space)

    data_config = config.get("data", {})
    rc_eval = bool(data_config.get("rc_eval", False))
    n_sequences: int | None = None
    if _standalone_sliding_window_enabled(config):
        overlap = float(data_config.get("sliding_window_overlap", 0.5))
        loader, n_sequences = make_sliding_window_dataloader(
            records=records,
            tokenizer=tokenizer,
            max_length=max_length,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
            num_workers=num_workers,
            overlap=overlap,
            order2idx=label_space["order2idx"],
            superfamily2idx=label_space["superfamily2idx"],
            rc_eval=rc_eval,
        )
        print(
            "External Repbase evaluation: using sliding-window inference "
            f"(overlap={overlap}, n_sequences={n_sequences}, "
            f"n_windows={len(loader.dataset)})"
        )
    else:
        dataset = TEDataset(
            records,
            tokenizer=tokenizer,
            max_length=max_length,
            pad_token_id=pad_token_id,
            augment=False,
            order2idx=label_space["order2idx"],
            superfamily2idx=label_space["superfamily2idx"],
            rc_eval=rc_eval,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        print(
            "External Repbase evaluation: using non-sliding inference "
            f"(n_sequences={len(records)})"
        )

    model.eval()
    model.to(device)

    if n_sequences is not None:
        accumulator = MeanLogitAccumulator(n_sequences, device="cpu")

        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            seq_idx = batch["seq_idx"].to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                order_logits, sf_logits = model(input_ids, attention_mask)

            accumulator.add(seq_idx, order_logits, sf_logits)

        _order_logits_mean, sf_logits_mean = accumulator.means()
        sf_logits_np = sf_logits_mean.numpy()
    else:
        logits_parts: list[np.ndarray] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                _order_logits, sf_logits = model(input_ids, attention_mask)

            logits_parts.append(sf_logits.float().cpu().numpy())
        sf_logits_np = np.concatenate(logits_parts, axis=0)

    sf_logits_tensor = torch.from_numpy(sf_logits_np)
    sf_probs_np = torch.softmax(sf_logits_tensor, dim=-1).numpy()
    sf_conf = sf_probs_np.max(axis=1)
    pred_idx = sf_probs_np.argmax(axis=1)

    rows: list[dict] = []
    for i, rec in enumerate(records):
        true_sf = rec["superfamily"]
        pred_sf = idx2sf[int(pred_idx[i])]
        true_order = rec["order"]
        pred_order = sf_to_order_map.get(pred_sf, pred_sf.split("/", 1)[0] if "/" in pred_sf else pred_sf)
        hier_conf = float(
            sf_probs_np[i, order_sf_indices.get(pred_order, [])].sum()
            if pred_order in order_sf_indices else 0.0
        )
        rows.append({
            "name": rec.get("name", ""),
            "species": rec.get("species", ""),
            "sequence_length": len(rec.get("sequence", "")),
            "original_category": rec.get("original_category", rec.get("category", "")),
            "repeatmasker_label": rec.get("repeatmasker_label", ""),
            "true_order": true_order,
            "true_superfamily": true_sf,
            "pred_order": pred_order,
            "pred_superfamily": pred_sf,
            "sf_conf": float(sf_conf[i]),
            "hier_conf": hier_conf,
            "correct_superfamily": bool(true_sf == pred_sf),
            "correct_order": bool(true_order == pred_order),
        })

    df = pd.DataFrame(rows)
    df.attrs["label_space"] = label_space["name"]
    return df


def summarize_external_repbase_predictions(
    predictions: pd.DataFrame,
    diagnostics: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build summary and per-class result tables for external Repbase eval."""
    true_sf = predictions["true_superfamily"].tolist()
    pred_sf = predictions["pred_superfamily"].tolist()
    true_order = predictions["true_order"].tolist()
    pred_order = predictions["pred_order"].tolist()

    label_space = get_label_space(predictions.attrs.get("label_space", "panteon30"))
    metrics = compute_panteon_metrics(true_sf, pred_sf, sf_to_order_map=label_space["sf_to_order"])
    order_accuracy = float(np.mean([t == p for t, p in zip(true_order, pred_order)]))

    summary_df = pd.DataFrame([{
        "sf_f1": metrics["sf_f1"],
        "order_f1": metrics["order_f1"],
        "accuracy": metrics["accuracy"],
        "order_accuracy": order_accuracy,
        "n_total": metrics["n_total"],
        "loaded_records": diagnostics["loaded_records"],
        "mapped_records": diagnostics["mapped_records"],
        "skipped_records": diagnostics["skipped_records"],
    }])

    labels = sorted(set(true_sf) | set(pred_sf))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_sf,
        pred_sf,
        labels=labels,
        zero_division=0,
    )
    per_class_rows = []
    for label, p, r, f, s in zip(labels, precision, recall, f1, support):
        pred_count = int((predictions["pred_superfamily"] == label).sum())
        per_class_rows.append({
            "superfamily": label,
            "order": label_space["sf_to_order"].get(label, "UNKNOWN"),
            "precision": float(p),
            "recall": float(r),
            "sf_f1": float(f),
            "support": int(s),
            "predicted_count": pred_count,
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # Confusion matrix
    cm_labels = metrics["confusion_matrix_labels"]
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=cm_labels, columns=cm_labels)

    return summary_df, per_class_df, cm_df


def write_external_repbase_diagnostics(
    diagnostics: dict,
    output_dir: Path,
) -> None:
    """Write parse/mapping diagnostics for external Repbase evaluation."""
    diagnostic_tables = {
        "repbase_raw_category_counts.csv": diagnostics["raw_category_counts"],
        "repbase_repeatmasker_counts.csv": diagnostics["repeatmasker_counts"],
        "repbase_panteon_superfamily_counts.csv": diagnostics["panteon_superfamily_counts"],
        "repbase_panteon_order_counts.csv": diagnostics["panteon_order_counts"],
        "repbase_skipped_reason_counts.csv": diagnostics["skipped_reason_counts"],
        "repbase_skipped_category_counts.csv": diagnostics["skipped_category_counts"],
        "repbase_skipped_repeatmasker_counts.csv": diagnostics["skipped_repeatmasker_counts"],
    }
    for filename, counts in diagnostic_tables.items():
        pd.DataFrame(
            [{"label": key, "count": value} for key, value in counts.items()]
        ).to_csv(output_dir / filename, index=False)

    pd.DataFrame(diagnostics["mapped_class_rows"]).to_csv(
        output_dir / "repbase_mapped_records.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"repeatmasker_label": rm, "panteon_superfamily": sf, "panteon_order": (SF_TO_ORDER.get(sf) or (sf.split("/", 1)[0] if "/" in sf else sf))}
            for rm, sf in diagnostics["mapping_rules"].items()
        ]
    ).to_csv(output_dir / "repbase_repeatmasker_to_panteon_rules.csv", index=False)


def evaluate_external_repbase_full(
    model,
    fasta_path: Path,
    toml_path: Path,
    tokenizer,
    device: torch.device,
    config: dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Repbase FASTA/REF input with a PanTEon-trained GLARE model.

    ``fasta_path`` may be a single REF/FASTA file or a directory containing
    sorted ``*.ref`` files.  Label-subset filtering is applied after all mapped
    records are loaded.
    """
    label_space = _label_space_from_config(config, default="panteon30")
    data_config = config.get("data", {})
    if label_space["name"] == "panteon30":
        records, diagnostics = load_external_repbase_fasta_for_panteon(
            fasta_path=fasta_path,
            toml_path=toml_path,
        )
    elif fasta_path.is_dir():
        records, diagnostics = load_repbase_dir_for_label_space(
            data_dir=fasta_path,
            toml_path=toml_path,
            label_space_name=str(label_space["name"]),
        )
    else:
        with open(toml_path, "rb") as fh:
            source_to_target: dict[str, str] = tomllib.load(fh)
        raw_records = parse_ref_file(fasta_path)
        records, diagnostics = map_records_to_configured_label_space(
            raw_records,
            source_to_target,
            label_space_name=str(label_space["name"]),
        )
        diagnostics["fasta_path"] = str(fasta_path)
        diagnostics["toml_path"] = str(toml_path)
        diagnostics["ref_files"] = 1

    if _repbase_stratified_random_split_enabled(config):
        split_config = data_config.get("repbase_split") or {}
        if isinstance(split_config, str):
            split_config = {"mode": split_config}
        val_fraction = float(config.get("training", {}).get("val_fraction", 0.1))
        test_fraction = float(
            split_config.get(
                "test_fraction",
                data_config.get("test_fraction", val_fraction),
            )
        )
        seed = int(config.get("training", {}).get("seed", 7))
        _rep_train, _rep_val, rep_test_records, rep_split_stats = split_stratified_train_val_test_records(
            records=records,
            val_fraction=val_fraction,
            seed=seed,
            superfamily2idx=label_space["superfamily2idx"],
            test_fraction=test_fraction,
            split_mode="stratified_random_repbase_eval_test",
        )
        diagnostics = _repbase_eval_diagnostics_from_records(
            rep_test_records,
            diagnostics,
            rep_split_stats,
        )
        records = rep_test_records
        print("Repbase stratified random test split:")
        print(format_split_stats(rep_split_stats))

    label_subset_labels, label_subset_stats = resolve_superfamily_label_subset(
        data_config
    )
    if label_subset_labels is not None:
        records, subset_filter_stats = filter_records_by_superfamilies(
            records,
            label_subset_labels,
            dataset_name="External Repbase mapped records",
        )
        diagnostics["label_subset"] = label_subset_stats
        diagnostics["label_subset_filter"] = subset_filter_stats
        print(
            "External Repbase label subset filter: "
            f"kept={subset_filter_stats['kept_records']}/"
            f"{subset_filter_stats['total_records']} "
            f"skipped={subset_filter_stats['skipped_records']} "
            f"labels={','.join(label_subset_labels)}"
        )
    if not records:
        raise ValueError(
            f"No records from {fasta_path} mapped into the PanTEon/GLARE 30-label space."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_external_repbase_diagnostics(diagnostics, output_dir)

    print(
        "External Repbase parse/mapping summary: "
        f"loaded={diagnostics['loaded_records']}, "
        f"mapped={diagnostics['mapped_records']}, "
        f"skipped={diagnostics['skipped_records']}"
    )

    predictions = predict_panteon_records(
        model=model,
        records=records,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )

    summary_df, per_class_df, cm_df = summarize_external_repbase_predictions(
        predictions=predictions,
        diagnostics=diagnostics,
    )

    predictions_path = output_dir / "repbase_predictions.csv"
    per_class_path = output_dir / "repbase_per_class_f1.csv"
    summary_path = output_dir / "repbase_summary.csv"
    cm_path = output_dir / "repbase_confusion_matrix.csv"
    predictions.to_csv(predictions_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    cm_df.to_csv(cm_path)

    # Save confusion matrix figure
    cm_arr = cm_df.values.astype(int)
    cm_labels = cm_df.index.tolist()
    n_labels = len(cm_labels)
    cm_norm = cm_arr.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm_norm / row_sums
    fig, ax = plt.subplots()
    fig.set_size_inches(max(8, n_labels * 0.5), max(6, n_labels * 0.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n_labels),
        yticks=range(n_labels),
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation="vertical")
    thresh = cm_norm.max() / 2.0
    for i in range(n_labels):
        for j in range(n_labels):
            ax.text(
                j, i, str(cm_arr[i, j]),
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=max(4, min(8, 80 // n_labels)),
            )
    fig.tight_layout()
    cm_fig_path = output_dir / "repbase_confusion_matrix.png"
    fig.savefig(cm_fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix figure → {cm_fig_path}")

    row = summary_df.iloc[0]
    print("\n=== External Repbase/PanTEon Results ===")
    print(f"  Superfamily weighted F1 : {row['sf_f1']:.4f}  ({row['sf_f1']*100:.2f}%)")
    print(f"  Order weighted F1       : {row['order_f1']:.4f}  ({row['order_f1']*100:.2f}%)")
    print(f"  Superfamily accuracy    : {row['accuracy']:.4f}  ({row['accuracy']*100:.2f}%)")
    print(f"  Order accuracy          : {row['order_accuracy']:.4f}  ({row['order_accuracy']*100:.2f}%)")
    print(f"  Loaded records          : {int(row['loaded_records'])}")
    print(f"  Mapped records          : {int(row['mapped_records'])}")
    print(f"  Skipped records         : {int(row['skipped_records'])}")
    print(f"\nPredictions saved to     : {predictions_path}")
    print(f"Per-class F1 saved to    : {per_class_path}")
    print(f"Summary saved to         : {summary_path}")
    print(f"Confusion matrix saved to: {cm_path}")

    return summary_df, per_class_df


# ---------------------------------------------------------------------------
# Per-sequence inference (works for both workflows)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_threshold(
    model,
    loader,
    device: torch.device,
    threshold: float = 0.7,
    use_hierarchical_confidence: bool = True,
    label_space: dict | None = None,
    temperature_scaler=None,
    mahal_detector=None,
    ood_method: str = "hier_conf",
) -> pd.DataFrame:
    """Run inference and apply confidence threshold.

    .. note::
        This function is kept for the **legacy Repbase workflow**.
        For PanTEon evaluation, use :func:`evaluate_panteon_test` instead
        (no threshold is applied there).

    Args:
        model: Trained :class:`~glare.model.GLAREModel`.
        loader: A ``(DataLoader, records)`` tuple **or** just a DataLoader.
            If a tuple, records are used for ground-truth labels.
        device: Target device.
        threshold: Confidence threshold; a sequence is *classified* only when
            its confidence meets this threshold.
        use_hierarchical_confidence: If ``True`` (default), use order-level
            confidence (sum of SF probs for the predicted order) as the
            threshold metric. If ``False``, use flat SF max confidence.
        temperature_scaler: Optional :class:`~glare.ood.TemperatureScaler`.
            When provided, logits are divided by T before computing softmax
            probabilities (so ``sf_conf`` and ``hier_conf`` use calibrated
            probabilities).
        mahal_detector: Optional fitted :class:`~glare.ood.MahalanobisOOD`.
            When provided, ``mahal_score`` is computed from trunk embeddings.
        ood_method: Which score to use for the ``classified`` flag.
            One of ``"hier_conf"`` (default, existing behaviour),
            ``"energy"`` (lower energy_score = classified),
            ``"mahal"`` (lower mahal_score = classified),
            ``"combined"`` (geometric mean of normalised energy + mahal).

    Returns:
        :class:`pandas.DataFrame` with columns:
        ``true_order``, ``true_superfamily``,
        ``pred_order``, ``pred_superfamily``,
        ``sf_conf``, ``hier_conf``, ``classified``,
        ``energy_score``, ``mahal_score``.
    """
    from .ood import EnergyScorer

    # Accept either a plain DataLoader or a (loader, records) tuple
    if isinstance(loader, tuple):
        loader, records = loader
    else:
        records = None

    model.eval()
    model.to(device)

    label_space = label_space or get_label_space("panteon30")
    idx2sf = label_space["idx2superfamily"]
    sf_to_order_map = label_space["sf_to_order"]
    order_sf_indices = _order_sf_indices_for_label_space(label_space)

    temperature = temperature_scaler.temperature if temperature_scaler is not None else 1.0
    has_trunk_extraction = (
        mahal_detector is not None and hasattr(model, "extract_trunk_embeddings")
    )

    all_rows: list[dict] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        sf_true = batch["superfamily_label"].cpu().numpy()

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            _order_logits, sf_logits = model(input_ids, attention_mask)

        sf_logits_f32 = sf_logits.float()

        # Apply temperature scaling before softmax
        if temperature_scaler is not None:
            sf_logits_for_probs = temperature_scaler.calibrate(sf_logits_f32)
        else:
            sf_logits_for_probs = sf_logits_f32

        sf_probs = torch.softmax(sf_logits_for_probs, dim=-1).cpu().numpy()   # (B, C_sf)
        sf_conf = sf_probs.max(axis=1)                               # (B,)
        sf_pred_idx = sf_probs.argmax(axis=1)                        # (B,)

        # Energy scores on raw logits
        energy_scores = EnergyScorer.score(sf_logits_f32.cpu(), T=temperature).numpy()

        # Compute hierarchical (order-level) confidence
        pred_orders_batch = [
            sf_to_order_map.get(idx2sf[int(i)], "")
            for i in sf_pred_idx
        ]
        hier_conf = np.array([
            sf_probs[b, order_sf_indices.get(pred_orders_batch[b], [])].sum()
            if pred_orders_batch[b] in order_sf_indices else 0.0
            for b in range(len(sf_pred_idx))
        ])

        # Mahalanobis scores (optional)
        if has_trunk_extraction:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                trunk_emb = model.extract_trunk_embeddings(input_ids, attention_mask)
            trunk_np = trunk_emb.float().cpu().numpy()
            mahal_scores = mahal_detector.score_batch(trunk_np)
        else:
            mahal_scores = np.full(input_ids.shape[0], np.nan)

        # Select which confidence/score to use for thresholding
        if ood_method == "hier_conf":
            conf_for_threshold = hier_conf if use_hierarchical_confidence else sf_conf
        elif ood_method == "energy":
            # Lower energy = more in-distribution → classified when energy <= threshold
            conf_for_threshold = -energy_scores  # negate so higher = more ID
        elif ood_method == "mahal":
            # Lower mahal = more in-distribution → classified when mahal <= threshold
            conf_for_threshold = -mahal_scores
        elif ood_method == "combined":
            # Geometric mean of normalised (negated) energy and mahal
            neg_energy = -energy_scores
            neg_mahal = -mahal_scores
            # Normalise to [0,1] range per batch (min-max); fall back to energy if mahal is NaN
            def _minmax(arr: np.ndarray) -> np.ndarray:
                lo, hi = arr.min(), arr.max()
                return (arr - lo) / (hi - lo + 1e-12)
            if not np.isnan(neg_mahal).any():
                conf_for_threshold = np.sqrt(_minmax(neg_energy) * _minmax(neg_mahal))
            else:
                conf_for_threshold = _minmax(neg_energy)
        else:
            conf_for_threshold = hier_conf if use_hierarchical_confidence else sf_conf

        batch_size_b = input_ids.shape[0]
        for i in range(batch_size_b):
            pred_sf = idx2sf[int(sf_pred_idx[i])]
            true_sf = idx2sf[int(sf_true[i])]
            classified = bool(conf_for_threshold[i] >= threshold)
            row: dict = {
                "true_order": sf_to_order_map.get(true_sf, true_sf),
                "true_superfamily": true_sf,
                "pred_order": sf_to_order_map.get(pred_sf, pred_sf),
                "pred_superfamily": pred_sf,
                "sf_conf": float(sf_conf[i]),
                "hier_conf": float(hier_conf[i]),
                "classified": classified,
                "energy_score": float(energy_scores[i]),
                "mahal_score": float(mahal_scores[i]),
            }
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# F1 at threshold (legacy)
# ---------------------------------------------------------------------------

def compute_f1_at_threshold(df: pd.DataFrame) -> dict:
    """Compute weighted F1 on the *classified* subset.

    Args:
        df: Output of :func:`predict_with_threshold`.

    Returns:
        Dict with keys:
        ``order_f1``, ``sf_f1``, ``pct_classified``, ``n_total``, ``n_classified``.
    """
    n_total = len(df)
    classified = df[df["classified"]]
    n_classified = len(classified)

    if n_classified == 0:
        return {
            "order_f1": 0.0,
            "sf_f1": 0.0,
            "pct_classified": 0.0,
            "n_total": n_total,
            "n_classified": 0,
        }

    order_f1 = float(
        f1_score(
            classified["true_order"],
            classified["pred_order"],
            average="weighted",
            zero_division=0,
        )
    )
    sf_f1 = float(
        f1_score(
            classified["true_superfamily"],
            classified["pred_superfamily"],
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "order_f1": order_f1,
        "sf_f1": sf_f1,
        "pct_classified": n_classified / n_total,
        "n_total": n_total,
        "n_classified": n_classified,
    }


# ---------------------------------------------------------------------------
# Legacy holdout group definitions
# ---------------------------------------------------------------------------

# LEGACY: Repbase holdout groups (filename → group name).
# Only used when evaluating with the old default.yaml / Repbase workflow.
HOLDOUT_GROUPS: dict[str, list[str]] = {
    "Fungi": ["fngrep.ref", "fugrep.ref"],
    "Sea_Anemone": ["nemrep.ref"],
    "Nematodes": ["celrep.ref", "cbrrep.ref"],
    "Grasses": ["repbase.ref", "mcotrep.ref"],
    "Tunicates": ["cinrep.ref", "cinunc.ref"],
}


def evaluate_holdout_groups(
    model,
    holdout_dir: Path,
    toml_path: Path,
    tokenizer,
    device: torch.device,
    config: dict,
    thresholds: list[float] | None = None,
    output_dir: Path | None = None,
    temperature_scaler=None,
    mahal_detector=None,
    ood_method: str = "hier_conf",
) -> pd.DataFrame:
    """Evaluate model on all 5 holdout groups at all thresholds (legacy Repbase).

    .. deprecated::
        This function is for the legacy Repbase workflow.  For PanTEon
        evaluation, use :func:`evaluate_panteon_full` instead.

    Args:
        model: Trained :class:`~glare.model.GLAREModel` (on CPU or GPU).
        holdout_dir: Directory containing ``.ref`` holdout files.
        toml_path: Path to ``repbase-to-repeatmasker.toml``.
        tokenizer: NTv3 tokenizer.
        device: Inference device.
        config: Full config dict (used for ``batch_size``, ``max_length``,
            and ``evaluation.use_hierarchical_confidence``).
        thresholds: List of confidence thresholds to evaluate; defaults to
            ``[0.7, 0.9]``.
        output_dir: If provided, saves ``glare_holdout_f1_results.csv`` there.
        temperature_scaler: Optional :class:`~glare.ood.TemperatureScaler` for
            post-hoc calibration.
        mahal_detector: Optional fitted :class:`~glare.ood.MahalanobisOOD`.
        ood_method: Which score to use for the ``classified`` flag.
            One of ``"hier_conf"`` (default), ``"energy"``, ``"mahal"``,
            ``"combined"``.

    Returns:
        :class:`pandas.DataFrame` matching the baseline holdout-results schema:
        columns ``group``, ``threshold``, ``order_f1``, ``sf_f1``,
        ``pct_classified``, ``n_total``, ``n_classified``.
    """
    from .data import make_holdout_dataloader

    if thresholds is None:
        thresholds = config.get("evaluation", {}).get("thresholds", [0.7, 0.9])

    use_hier = config.get("evaluation", {}).get("use_hierarchical_confidence", True)
    label_space = _label_space_from_config(config, default="legacy45")
    idx2sf = label_space["idx2superfamily"]
    sf_to_order_map = label_space["sf_to_order"]
    order_sf_indices = _order_sf_indices_for_label_space(label_space)

    batch_size = config["training"]["batch_size"]
    max_length = config["model"]["max_length"]

    model.eval()
    model.to(device)

    result_rows: list[dict] = []

    for group_name, filenames in HOLDOUT_GROUPS.items():
        # Collect paths for files that actually exist
        group_files = [
            holdout_dir / fn for fn in filenames if (holdout_dir / fn).exists()
        ]
        if not group_files:
            continue

        loader, _ = make_holdout_dataloader(
            holdout_files=group_files,
            toml_path=toml_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=2,
        )

        # Run inference once (collect all probs)
        all_sf_probs: list[np.ndarray] = []
        all_sf_true: list[int] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    _order_logits, sf_logits = model(input_ids, attention_mask)

                all_sf_probs.append(
                    torch.softmax(sf_logits.float(), dim=-1).cpu().numpy()
                )
                all_sf_true.extend(batch["superfamily_label"].numpy().tolist())

        sf_probs_np = np.concatenate(all_sf_probs, axis=0)   # (N, C_sf)
        sf_conf = sf_probs_np.max(axis=1)
        sf_pred_idx = sf_probs_np.argmax(axis=1)

        # Derive string labels — order from SF via SF_TO_ORDER lookup
        true_sf_labels = [idx2sf[i] for i in all_sf_true]
        pred_sf_labels = [idx2sf[int(i)] for i in sf_pred_idx]
        true_order_labels = [sf_to_order_map.get(s, s) for s in true_sf_labels]
        pred_order_labels = [sf_to_order_map.get(s, s) for s in pred_sf_labels]

        # Compute hierarchical (order-level) confidence
        hier_conf = np.array([
            sf_probs_np[b, order_sf_indices.get(pred_order_labels[b], [])].sum()
            if pred_order_labels[b] in order_sf_indices else 0.0
            for b in range(len(sf_pred_idx))
        ])

        # Select which confidence to use for thresholding
        conf_for_threshold = hier_conf if use_hier else sf_conf

        for threshold in thresholds:
            classified_mask = conf_for_threshold >= threshold
            n_total = len(all_sf_true)
            n_classified = int(classified_mask.sum())

            if n_classified == 0:
                row = {
                    "group": group_name,
                    "threshold": threshold,
                    "order_f1": 0.0,
                    "sf_f1": 0.0,
                    "pct_classified": 0.0,
                    "n_total": n_total,
                    "n_classified": 0,
                }
            else:
                t_ord = [true_order_labels[i] for i, m in enumerate(classified_mask) if m]
                p_ord = [pred_order_labels[i] for i, m in enumerate(classified_mask) if m]
                t_sf = [true_sf_labels[i] for i, m in enumerate(classified_mask) if m]
                p_sf = [pred_sf_labels[i] for i, m in enumerate(classified_mask) if m]

                order_f1 = float(f1_score(t_ord, p_ord, average="weighted", zero_division=0))
                sf_f1 = float(f1_score(t_sf, p_sf, average="weighted", zero_division=0))

                row = {
                    "group": group_name,
                    "threshold": threshold,
                    "order_f1": order_f1,
                    "sf_f1": sf_f1,
                    "pct_classified": n_classified / n_total,
                    "n_total": n_total,
                    "n_classified": n_classified,
                }

            result_rows.append(row)

    results_df = pd.DataFrame(result_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "glare_holdout_f1_results.csv"
        results_df.to_csv(out_path, index=False)

    return results_df
