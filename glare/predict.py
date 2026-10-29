"""Inference on new sequences (FASTA or .ref files).

Design: dual-head model, SF-primary.
  - The model outputs both order logits (auxiliary) and superfamily logits (primary).
  - At inference time, ``pred_order`` is derived from ``pred_superfamily`` via
    :func:`~glare.constants.sf_to_order` (splitting on ``"/"``).
  - Classification threshold is applied to the flat superfamily confidence
    (``sf_conf``) by default.  Hierarchical order confidence can be requested
    with ``use_hierarchical_confidence=True``:
    ``order_conf = sum of SF probs for all SFs in the predicted order``.
"""

from __future__ import annotations

import zlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from Bio import SeqIO

from .constants import IDX2SUPERFAMILY, SUPERFAMILY_LABELS, sf_to_order, get_label_space
from .data import SlidingWindowInferenceDataset
from .inference import MeanLogitAccumulator


# ---------------------------------------------------------------------------
# Pre-compute order → SF indices mapping (module-level, computed once)
# ---------------------------------------------------------------------------

# Maps each order name to the list of SF indices belonging to that order.
# Used for hierarchical confidence: order_conf = sum(sf_probs[ORDER_SF_INDICES[order]])
ORDER_SF_INDICES: dict[str, list[int]] = {}
for _idx, _sf in enumerate(SUPERFAMILY_LABELS):
    _order = sf_to_order(_sf)
    if _order not in ORDER_SF_INDICES:
        ORDER_SF_INDICES[_order] = []
    ORDER_SF_INDICES[_order].append(_idx)


UNKNOWN_CLASS_LABEL = "UNKNOWN"

_PREDICTION_ORDER_COLOR_NAMES = {
    "LTR": "tab:blue",
    "TIR": "tab:orange",
    "LINE": "tab:green",
    "SINE": "tab:red",
    "HELITRON": "tab:purple",
    "DIRS": "tab:brown",
    "PLE": "tab:pink",
    "CRYPTON": "tab:olive",
    "MAVERICK": "tab:cyan",
    "SATELLITE": "goldenrod",
    "STRUCTURAL_RNA": "steelblue",
}

PREDICTION_OUTPUT_COLUMNS = [
    "id",
    "pred_order",
    "pred_superfamily",
    "repeatmasker_class",
    "sf_conf",
    "is_unknown",
]


def load_label_space_metadata(labels_path: Path) -> dict:
    """Load training-time label metadata from a bundle ``labels.json`` file."""
    with open(labels_path) as fh:
        metadata = json.load(fh)
    if not isinstance(metadata, dict):
        raise ValueError(f"Label metadata in {labels_path} must be a JSON object.")
    if "idx2superfamily" not in metadata and "superfamily_labels" not in metadata:
        raise ValueError(
            f"Label metadata in {labels_path} must contain idx2superfamily or "
            "superfamily_labels."
        )
    return metadata


def _label_space_mappings(label_space: Optional[str | dict]) -> tuple[dict[int, str], dict[str, str], dict[str, list[int]]]:
    """Return prediction decoding maps from constants or serialized metadata."""
    if isinstance(label_space, dict):
        raw_idx2sf = label_space.get("idx2superfamily")
        if raw_idx2sf:
            idx2sf = {int(idx): str(sf) for idx, sf in dict(raw_idx2sf).items()}
        else:
            labels = [str(label) for label in label_space.get("superfamily_labels", [])]
            idx2sf = {idx: label for idx, label in enumerate(labels)}
        raw_sf_to_order = label_space.get("sf_to_order") or {}
        sf_to_order_map = {str(sf): str(order) for sf, order in dict(raw_sf_to_order).items()}
        missing_orders = sorted(set(idx2sf.values()) - set(sf_to_order_map))
        if missing_orders:
            raise ValueError(
                "Serialized label metadata is missing sf_to_order entries for "
                f"superfamilies: {missing_orders[:10]}"
                f"{'...' if len(missing_orders) > 10 else ''}"
            )
    elif label_space is not None:
        ls = get_label_space(label_space)
        idx2sf = {int(idx): str(sf) for idx, sf in dict(ls["idx2superfamily"]).items()}
        sf_to_order_map = {str(sf): str(order) for sf, order in dict(ls["sf_to_order"]).items()}
    else:
        idx2sf = dict(IDX2SUPERFAMILY)
        sf_to_order_map = {sf: sf_to_order(sf) for sf in idx2sf.values()}

    order_sf_indices: dict[str, list[int]] = {}
    for idx, sf in idx2sf.items():
        order = sf_to_order_map[sf]
        if order not in order_sf_indices:
            order_sf_indices[order] = []
        order_sf_indices[order].append(idx)
    return idx2sf, sf_to_order_map, order_sf_indices

SUMMARY_OUTPUT_COLUMNS = [
    "pred_order",
    "pred_superfamily",
    "repeatmasker_class",
    "count",
    "percent",
    "classified_count",
    "mean_sf_conf",
]


# ---------------------------------------------------------------------------
# FASTA loader
# ---------------------------------------------------------------------------

def _load_fasta_records(
    fasta_path: Path,
    record_range: Optional[tuple[int, int]] = None,
    input_index_offset: int = 0,
) -> list[dict]:
    """Load an optional contiguous range of FASTA records via BioPython."""
    start, stop = record_range if record_range is not None else (0, None)
    records = []
    with open(fasta_path, "r", errors="replace") as fasta_handle:
        for input_index, seq_rec in enumerate(SeqIO.parse(fasta_handle, "fasta")):
            if input_index < start:
                continue
            if stop is not None and input_index >= stop:
                break
            records.append(
                {
                    "id": seq_rec.id,
                    "sequence": str(seq_rec.seq).upper(),
                    "input_index": input_index + input_index_offset,
                }
            )
    return records


def _iter_ref_records(ref_path: Path):
    """Yield Repbase records in source order without retaining the whole input."""
    current_id: str | None = None
    seq_parts: list[str] = []

    with open(ref_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_id is not None:
                    yield {"id": current_id, "sequence": "".join(seq_parts).upper()}
                parts = line[1:].split("\t")
                current_id = parts[0].strip()
                seq_parts = []
            else:
                stripped = line.strip()
                if stripped:
                    seq_parts.append(stripped)

    if current_id is not None:
        yield {"id": current_id, "sequence": "".join(seq_parts).upper()}


def _load_ref_records(
    ref_path: Path,
    record_range: Optional[tuple[int, int]] = None,
    input_index_offset: int = 0,
) -> list[dict]:
    """Load an optional contiguous range from a Repbase ``.ref`` file."""
    start, stop = record_range if record_range is not None else (0, None)
    records = []
    for input_index, record in enumerate(_iter_ref_records(ref_path)):
        if input_index < start:
            continue
        if stop is not None and input_index >= stop:
            break
        records.append({**record, "input_index": input_index + input_index_offset})
    return records


def count_sequence_records(input_path: Path) -> int:
    """Count ordered FASTA/REF records without materializing their sequences."""
    return len(sequence_record_offsets(input_path))


def sequence_record_offsets(input_path: Path) -> list[int]:
    """Index FASTA/REF header byte offsets in one streaming source pass."""
    offsets: list[int] = []
    with open(input_path, "rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if line.startswith(b">"):
                offsets.append(offset)
    return offsets


def prepare_sequence_shards(
    input_path: Path,
    ranges: list[tuple[int, int]],
    output_dir: Path,
    record_offsets: Optional[list[int]] = None,
) -> list[Path]:
    """Create exact contiguous shards from a one-pass header-offset index."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not ranges:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix or ".fasta"
    final_paths = [output_dir / f"input-{index:04d}{suffix}" for index in range(len(ranges))]
    temporary_paths = [path.with_suffix(path.suffix + ".tmp") for path in final_paths]
    offsets = record_offsets if record_offsets is not None else sequence_record_offsets(input_path)
    file_size = input_path.stat().st_size
    try:
        expected_count = ranges[-1][1]
        if len(offsets) != expected_count:
            raise RuntimeError(
                "Input changed while prediction shards were prepared: "
                f"expected {expected_count} records, observed {len(offsets)}."
            )
        with open(input_path, "rb") as source:
            for (start, stop), temporary_path in zip(ranges, temporary_paths):
                byte_start = offsets[start]
                byte_stop = offsets[stop] if stop < len(offsets) else file_size
                source.seek(byte_start)
                remaining = byte_stop - byte_start
                with open(temporary_path, "wb") as destination:
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError("Input ended while a prediction shard was copied.")
                        destination.write(chunk)
                        remaining -= len(chunk)
    except BaseException:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise
    for temporary_path, final_path in zip(temporary_paths, final_paths):
        os.replace(temporary_path, final_path)
    return final_paths


def empty_prediction_frame(include_input_index: bool = False) -> pd.DataFrame:
    """Return an empty full-result frame with the prediction schema."""
    columns = [
        "id", "pred_order", "pred_superfamily", "raw_pred_order",
        "raw_pred_superfamily", "repeatmasker_class", "raw_repeatmasker_class",
        "sf_conf", "hier_conf", "classified", "energy_score", "temperature",
        "mahal_score", "is_unknown",
    ]
    if include_input_index:
        columns.insert(0, "input_index")
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Prediction-output helpers
# ---------------------------------------------------------------------------


def _clean_class_value(value: object, unknown_label: str = UNKNOWN_CLASS_LABEL) -> str:
    """Return a display-safe class token, replacing missing values with UNKNOWN."""
    if value is None:
        return unknown_label
    try:
        if pd.isna(value):
            return unknown_label
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return unknown_label
    if text.lower() in {"unknown", "unclassified"}:
        return unknown_label
    return text


def repeatmasker_class_label(
    pred_order: object,
    pred_superfamily: object,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> str:
    """Format a prediction as ``ORDER/SUPERFAMILY`` for FASTA headers.

    Some GLARE label spaces already store superfamilies as RepeatMasker-style
    compound labels (for example ``LTR/GYPSY``), while PanTEon-style labels store
    only the terminal superfamily token (for example ``GYPSY``).  This helper
    avoids duplicating the order for compound labels and gracefully fills missing
    order/superfamily values with ``UNKNOWN``.
    """
    order = _clean_class_value(pred_order, unknown_label=unknown_label)
    superfamily = _clean_class_value(pred_superfamily, unknown_label=unknown_label)

    if "/" in superfamily:
        sf_order, sf_tail = superfamily.split("/", 1)
        if order == unknown_label or sf_order.upper() == order.upper():
            return f"{_clean_class_value(sf_order, unknown_label)}/{_clean_class_value(sf_tail, unknown_label)}"

    return f"{order}/{superfamily}"


def _terminal_superfamily(
    pred_order: object,
    pred_superfamily: object,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> str:
    """Return only the terminal superfamily token for summary tables."""
    order = _clean_class_value(pred_order, unknown_label=unknown_label)
    superfamily = _clean_class_value(pred_superfamily, unknown_label=unknown_label)
    if "/" in superfamily:
        sf_order, sf_tail = superfamily.split("/", 1)
        if order == unknown_label or sf_order.upper() == order.upper():
            return _clean_class_value(sf_tail, unknown_label=unknown_label)
    return superfamily


def _as_bool(value: object, default: bool = False) -> bool:
    """Coerce common scalar boolean representations without treating "False" as true."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return default


def _boolean_series(values: pd.Series, default: bool = False) -> pd.Series:
    """Coerce a series with possible string booleans into real booleans."""
    return values.map(lambda value: _as_bool(value, default=default)).astype(bool)


def _confidence_for_threshold(
    sf_conf: np.ndarray,
    hier_conf: np.ndarray,
    use_hierarchical_confidence: bool = False,
) -> np.ndarray:
    """Return the confidence score vector used for unknown thresholding.

    The default threshold basis is the flat superfamily confidence (``sf_conf``).
    Passing ``use_hierarchical_confidence=True`` preserves the explicit option to
    threshold on hierarchical order confidence (``hier_conf``).
    """
    return hier_conf if use_hierarchical_confidence else sf_conf


def _unknown_prediction_mask(predictions: pd.DataFrame) -> pd.Series:
    """Return rows that should be displayed as explicit UNKNOWN predictions."""
    mask = pd.Series(False, index=predictions.index)
    if "is_unknown" in predictions:
        mask = mask | _boolean_series(predictions["is_unknown"], default=False)
    if "classified" in predictions:
        mask = mask | ~_boolean_series(predictions["classified"], default=True)
    return mask


def apply_unknown_prediction_labels(
    predictions: pd.DataFrame,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
    preserve_raw: bool = True,
) -> pd.DataFrame:
    """Apply user-facing UNKNOWN labels to unknown/unclassified predictions.

    Rows flagged by ``is_unknown`` or rows with ``classified == False`` are
    represented as ``UNKNOWN/UNKNOWN`` in user-facing predicted class columns.
    When ``preserve_raw`` is true, the original model-predicted class labels are
    retained in ``raw_pred_order``, ``raw_pred_superfamily``, and
    ``raw_repeatmasker_class`` columns before the user-facing columns are
    replaced.
    """
    labeled = predictions.copy()

    if "pred_order" not in labeled:
        labeled["pred_order"] = unknown_label
    if "pred_superfamily" not in labeled:
        labeled["pred_superfamily"] = unknown_label

    labeled["pred_order"] = labeled["pred_order"].map(
        lambda value: _clean_class_value(value, unknown_label=unknown_label)
    )
    labeled["pred_superfamily"] = labeled["pred_superfamily"].map(
        lambda value: _clean_class_value(value, unknown_label=unknown_label)
    )

    if preserve_raw:
        if "raw_pred_order" not in labeled:
            labeled["raw_pred_order"] = labeled["pred_order"]
        if "raw_pred_superfamily" not in labeled:
            labeled["raw_pred_superfamily"] = labeled["pred_superfamily"]
        labeled["raw_repeatmasker_class"] = [
            repeatmasker_class_label(order, sf, unknown_label=unknown_label)
            for order, sf in zip(labeled["raw_pred_order"], labeled["raw_pred_superfamily"])
        ]

    unknown_mask = _unknown_prediction_mask(labeled)
    labeled.loc[unknown_mask, ["pred_order", "pred_superfamily"]] = unknown_label
    labeled["repeatmasker_class"] = [
        repeatmasker_class_label(order, sf, unknown_label=unknown_label)
        for order, sf in zip(labeled["pred_order"], labeled["pred_superfamily"])
    ]

    return labeled


def prediction_output_frame(
    predictions: pd.DataFrame,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> pd.DataFrame:
    """Return the slim, user-facing prediction table for CSV output."""
    labeled = apply_unknown_prediction_labels(
        predictions,
        unknown_label=unknown_label,
        preserve_raw=False,
    )
    for column in PREDICTION_OUTPUT_COLUMNS:
        if column not in labeled:
            labeled[column] = np.nan
    return labeled[PREDICTION_OUTPUT_COLUMNS]


def build_prediction_summary(
    predictions: pd.DataFrame,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> pd.DataFrame:
    """Aggregate per-sequence predictions by predicted order and superfamily.

    The returned table is intentionally data-visualization friendly: each row is
    one predicted ``ORDER/SUPERFAMILY`` class with counts, percentages, the
    classified-call count, and mean flat superfamily confidence when available.
    """
    columns = SUMMARY_OUTPUT_COLUMNS
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    summary_source = apply_unknown_prediction_labels(
        predictions,
        unknown_label=unknown_label,
        preserve_raw=False,
    )
    summary_source["pred_superfamily"] = [
        _terminal_superfamily(order, sf, unknown_label=unknown_label)
        for order, sf in zip(
            summary_source["pred_order"], summary_source["pred_superfamily"]
        )
    ]
    summary_source["repeatmasker_class"] = [
        repeatmasker_class_label(order, sf, unknown_label=unknown_label)
        for order, sf in zip(
            summary_source["pred_order"], summary_source["pred_superfamily"]
        )
    ]

    total = len(summary_source)
    grouped = summary_source.groupby(
        ["pred_order", "pred_superfamily", "repeatmasker_class"],
        dropna=False,
        sort=True,
    )

    rows: list[dict] = []
    for (order, superfamily, rm_class), group in grouped:
        row = {
            "pred_order": order,
            "pred_superfamily": superfamily,
            "repeatmasker_class": rm_class,
            "count": int(len(group)),
            "percent": round((len(group) / total) * 100.0, 4) if total else 0.0,
            "classified_count": int(group["classified"].fillna(False).astype(bool).sum())
            if "classified" in group
            else np.nan,
            "mean_sf_conf": float(group["sf_conf"].mean()) if "sf_conf" in group else np.nan,
        }
        rows.append(row)

    summary = pd.DataFrame(rows, columns=columns)
    return summary.sort_values(
        ["count", "pred_order", "pred_superfamily"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _prediction_order_color_map(
    orders: list[str] | pd.Index,
    get_cmap,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> dict[str, tuple[float, float, float, float]]:
    """Return stable base hues for predicted orders by label."""
    import matplotlib.colors as mcolors

    ordered_labels: list[str] = []
    for order in orders:
        order_text = _clean_class_value(order, unknown_label=unknown_label)
        if order_text not in ordered_labels:
            ordered_labels.append(order_text)

    fallback_palette = get_cmap("tab20", 20)
    color_by_order: dict[str, tuple[float, float, float, float]] = {}
    for order_text in ordered_labels:
        if order_text == unknown_label:
            color_by_order[order_text] = (0.08, 0.08, 0.08, 1.0)
        elif order_text in _PREDICTION_ORDER_COLOR_NAMES:
            color_by_order[order_text] = mcolors.to_rgba(_PREDICTION_ORDER_COLOR_NAMES[order_text])
        else:
            fallback_index = zlib.crc32(order_text.encode("utf-8")) % fallback_palette.N
            color_by_order[order_text] = mcolors.to_rgba(fallback_palette(fallback_index))
    return color_by_order


def _prediction_superfamily_shade(
    base_color: tuple[float, float, float, float],
    rank: int,
    n_shades: int,
) -> tuple[float, float, float, float]:
    """Return an abundance-ranked dark-to-light shade of a base order hue."""
    import colorsys
    import matplotlib.colors as mcolors

    red, green, blue = mcolors.to_rgb(base_color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    if n_shades <= 1:
        target_lightness = max(0.34, min(lightness, 0.56))
    else:
        darkest = 0.32
        lightest = 0.72
        target_lightness = darkest + (lightest - darkest) * (rank / (n_shades - 1))
    shaded_red, shaded_green, shaded_blue = colorsys.hls_to_rgb(
        hue,
        target_lightness,
        max(saturation, 0.45),
    )
    return (shaded_red, shaded_green, shaded_blue, base_color[3])


def _prediction_class_color_map(
    summary: pd.DataFrame,
    get_cmap,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float, float]],
    list[tuple[str, str]],
]:
    """Return shared per-order/per-superfamily colors and legend order.

    Orders receive stable label-based base hues.  Superfamilies within each order
    receive abundance-ranked shades of that hue, with the most abundant class
    assigned the darkest shade so the bottom stacked segment and the matching
    non-stacked bar use the same color.
    """
    required_columns = {"pred_order", "pred_superfamily", "count"}
    if summary.empty or not required_columns.issubset(summary.columns):
        return {}, []

    plot_df = summary.copy()
    plot_df["pred_order"] = plot_df["pred_order"].map(
        lambda value: _clean_class_value(value, unknown_label=unknown_label)
    )
    plot_df["pred_superfamily"] = plot_df["pred_superfamily"].map(
        lambda value: _clean_class_value(value, unknown_label=unknown_label)
    )
    plot_df["count"] = pd.to_numeric(plot_df["count"], errors="coerce").fillna(0).astype(int)
    plot_df = plot_df.loc[plot_df["count"] > 0].copy()
    if plot_df.empty:
        return {}, []

    grouped = plot_df.groupby(["pred_order", "pred_superfamily"], sort=True, dropna=False)["count"].sum()
    order_totals = grouped.groupby(level=0, sort=True).sum().sort_values(
        ascending=False,
        kind="mergesort",
    )
    orders = [str(order) for order in order_totals.index]
    color_by_order = _prediction_order_color_map(orders, get_cmap, unknown_label=unknown_label)

    color_by_class: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    class_order: list[tuple[str, str]] = []
    for order in orders:
        observed = grouped.loc[order].sort_values(ascending=False, kind="mergesort")
        superfamilies = [str(superfamily) for superfamily in observed.index]
        for rank, superfamily in enumerate(superfamilies):
            class_key = (order, superfamily)
            class_order.append(class_key)
            if order == unknown_label and superfamily == unknown_label:
                color_by_class[class_key] = (0.08, 0.08, 0.08, 1.0)
            else:
                base_color = color_by_order.get(order, (0.75, 0.75, 0.75, 1.0))
                color_by_class[class_key] = _prediction_superfamily_shade(
                    base_color,
                    rank,
                    len(superfamilies),
                )

    return color_by_class, class_order


def _prediction_order_sequence(
    summary: pd.DataFrame,
    unknown_label: str = UNKNOWN_CLASS_LABEL,
) -> list[str]:
    """Return abundance-ranked predicted orders using stacked-plot ordering."""
    if not {"pred_order", "count"}.issubset(summary.columns):
        return []

    plot_df = summary.copy()
    plot_df["pred_order"] = plot_df["pred_order"].map(
        lambda value: _clean_class_value(value, unknown_label=unknown_label)
    )
    plot_df["count"] = pd.to_numeric(plot_df["count"], errors="coerce").fillna(0).astype(int)
    plot_df = plot_df.loc[plot_df["count"] > 0]
    if plot_df.empty:
        return []

    order_totals = plot_df.groupby("pred_order", sort=True, dropna=False)["count"].sum()
    order_totals = order_totals.sort_values(ascending=False, kind="mergesort")
    return [str(order) for order in order_totals.index]


def plot_prediction_summary(
    summary: pd.DataFrame,
    output_path: Path,
    title: str = "Predicted TE class composition",
) -> None:
    """Write a publication-friendly class-composition bar plot from a summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    plot_df = plot_df.sort_values(
        ["count", "repeatmasker_class"], ascending=[True, True], kind="mergesort"
    )
    n_classes = len(plot_df)
    fig_height = max(4.0, min(18.0, 0.35 * n_classes + 1.5))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    plot_df["pred_order"] = plot_df["pred_order"].map(_clean_class_value)
    plot_df["pred_superfamily"] = plot_df["pred_superfamily"].map(_clean_class_value)
    color_by_class, class_order = _prediction_class_color_map(plot_df, plt.get_cmap)
    colors = [
        color_by_class.get(
            (str(row["pred_order"]), str(row["pred_superfamily"])),
            (0.75, 0.75, 0.75, 1.0),
        )
        for _, row in plot_df.iterrows()
    ]

    ax.barh(plot_df["repeatmasker_class"], plot_df["count"], color=colors, edgecolor="black", linewidth=0.4)
    max_count = max(int(plot_df["count"].max()), 1)
    for y_idx, (_, row) in enumerate(plot_df.iterrows()):
        label = f"{int(row['count'])} ({float(row['percent']):.1f}%)"
        ax.text(
            int(row["count"]) + max_count * 0.01,
            y_idx,
            label,
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Predicted sequence count")
    ax.set_ylabel("Predicted RepeatMasker class")
    ax.set_title(title)
    ax.set_xlim(0, max_count * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=color_by_class[(order, superfamily)],
            edgecolor="black",
            label=repeatmasker_class_label(order, superfamily),
        )
        for order, superfamily in class_order
    ]
    ax.legend(
        handles=legend_handles,
        title="Predicted class",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_stacked_prediction_summary(
    summary: pd.DataFrame,
    output_path: Path,
    title: str = "Predicted TE class composition by order and superfamily",
) -> None:
    """Write a stacked count bar plot from a prediction summary table.

    The plot uses one bar per ``pred_order`` and stacks each bar by
    ``pred_superfamily``.  Counts drive the bar heights, while labels include
    both counts and whole-run percentages so rare and ``UNKNOWN`` categories
    remain explicit without changing the summary schema.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"pred_order", "pred_superfamily", "count"}
    plot_df = summary.copy()
    if plot_df.empty or not required_columns.issubset(plot_df.columns):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    plot_df["pred_order"] = plot_df["pred_order"].map(_clean_class_value)
    plot_df["pred_superfamily"] = plot_df["pred_superfamily"].map(_clean_class_value)
    plot_df["count"] = pd.to_numeric(plot_df["count"], errors="coerce").fillna(0).astype(int)
    plot_df = plot_df.loc[plot_df["count"] > 0].copy()

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    grouped = plot_df.groupby(["pred_order", "pred_superfamily"], sort=True, dropna=False)["count"].sum()
    pivot = grouped.unstack(fill_value=0)
    order_totals = pivot.sum(axis=1).sort_values(ascending=False, kind="mergesort")
    sf_totals = pivot.sum(axis=0).sort_values(ascending=False, kind="mergesort")
    pivot = pivot.loc[order_totals.index, sf_totals.index]

    total_count = int(pivot.to_numpy().sum())
    n_orders = len(pivot.index)
    n_superfamilies = len(pivot.columns)
    fig_width = max(8.5, min(18.0, 1.0 * n_orders + 4.0))
    fig_height = max(5.0, min(14.0, 0.24 * n_superfamilies + 4.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    color_by_class, _class_order = _prediction_class_color_map(plot_df, plt.get_cmap)
    legend_handles: list[Patch] = []
    for order in pivot.index:
        order_text = str(order)
        observed = pivot.loc[order]
        observed = observed.loc[observed > 0].sort_values(ascending=False, kind="mergesort")
        for superfamily in observed.index:
            superfamily_text = str(superfamily)
            color = color_by_class.get(
                (order_text, superfamily_text),
                (0.75, 0.75, 0.75, 1.0),
            )
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    label=repeatmasker_class_label(order_text, superfamily_text),
                )
            )

    bottom = np.zeros(n_orders, dtype=float)
    x_positions = np.arange(n_orders)
    max_total = max(int(order_totals.max()), 1)

    for x_idx, order in enumerate(pivot.index):
        order_text = str(order)
        observed = pivot.loc[order]
        observed = observed.loc[observed > 0].sort_values(ascending=False, kind="mergesort")
        for superfamily, count_value in observed.items():
            superfamily_text = str(superfamily)
            count = float(count_value)
            color = color_by_class.get(
                (order_text, superfamily_text),
                (0.75, 0.75, 0.75, 1.0),
            )
            ax.bar(
                x_positions[x_idx],
                count,
                bottom=bottom[x_idx],
                color=color,
                edgecolor="black",
                linewidth=0.35,
            )
            percent = (count / total_count) * 100.0 if total_count else 0.0
            if count >= max_total * 0.08 or percent >= 3.0:
                ax.text(
                    x_idx,
                    bottom[x_idx] + count / 2.0,
                    f"{int(count)}\n({percent:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
            bottom[x_idx] += count

    for x_idx, total in enumerate(order_totals.to_numpy(dtype=float)):
        percent = (total / total_count) * 100.0 if total_count else 0.0
        ax.text(
            x_idx,
            total + max_total * 0.02,
            f"{int(total)} ({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(order) for order in pivot.index], rotation=35, ha="right")
    ax.set_xlabel("Predicted order")
    ax.set_ylabel("Predicted sequence count")
    ax.set_title(title)
    ax.set_ylim(0, max_total * 1.16)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles,
        title="Predicted class",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _load_sequence_records_for_output(input_path: Path) -> list[dict]:
    """Load input sequences for annotated FASTA output without changing bases."""
    records: list[dict] = []
    current_id: str | None = None
    seq_parts: list[str] = []
    suffix = input_path.suffix.lower()

    with open(input_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if current_id is not None:
                    records.append({"id": current_id, "sequence": "".join(seq_parts)})
                header = line[1:].strip()
                if suffix == ".ref":
                    current_id = header.split("\t", 1)[0].strip()
                else:
                    current_id = header.split(None, 1)[0].strip()
                seq_parts = []
            else:
                stripped = line.strip()
                if stripped:
                    seq_parts.append(stripped)

    if current_id is not None:
        records.append({"id": current_id, "sequence": "".join(seq_parts)})

    return records


def write_annotated_fasta(
    input_path: Path,
    predictions: pd.DataFrame,
    output_path: Path,
    line_width: int = 80,
) -> None:
    """Write input sequences with ``>ID#ORDER/SUPERFAMILY`` prediction headers."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_sequence_records_for_output(input_path)
    if len(records) != len(predictions):
        raise ValueError(
            f"Input sequence count ({len(records)}) does not match prediction count "
            f"({len(predictions)}); refusing to write misaligned annotated FASTA."
        )

    with open(output_path, "w") as out_fh:
        for record, (_, pred) in zip(records, predictions.iterrows()):
            base_id = _clean_class_value(record["id"].split("#", 1)[0])
            if _as_bool(pred.get("is_unknown", False)) or not _as_bool(pred.get("classified", True), default=True):
                rm_class = repeatmasker_class_label(UNKNOWN_CLASS_LABEL, UNKNOWN_CLASS_LABEL)
            else:
                rm_class = repeatmasker_class_label(
                    pred.get("pred_order", UNKNOWN_CLASS_LABEL),
                    pred.get("pred_superfamily", UNKNOWN_CLASS_LABEL),
                )
            out_fh.write(f">{base_id}#{rm_class}\n")
            sequence = record["sequence"]
            if line_width and line_width > 0:
                for start in range(0, len(sequence), line_width):
                    out_fh.write(sequence[start : start + line_width] + "\n")
            else:
                out_fh.write(sequence + "\n")


# ---------------------------------------------------------------------------
# Public inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_fasta(
    model,
    fasta_path: Path,
    tokenizer,
    device: torch.device,
    threshold: float = 0.7,
    batch_size: int = 32,
    num_workers: int = 2,
    max_length: int = 1024,
    use_hierarchical_confidence: bool = False,
    temperature_scaler: Optional["TemperatureScaler"] = None,  # type: ignore[name-defined]
    mahal_detector: Optional["MahalanobisOOD"] = None,  # type: ignore[name-defined]
    mahal_threshold: Optional[float] = None,
    label_space: Optional[str | dict] = None,
    record_range: Optional[tuple[int, int]] = None,
    include_input_index: bool = False,
    input_index_offset: int = 0,
    ood_method: str = "none",
    timing_metrics: Optional[dict] = None,
    sliding_window_overlap: Optional[float] = 0.5,
) -> pd.DataFrame:
    """Predict TE classifications for sequences in a FASTA or ``.ref`` file.

    Args:
        model: Trained :class:`~glare.model.GLAREModel`.
        fasta_path: Path to input ``.fasta`` / ``.fa`` / ``.ref`` file.
        tokenizer: NTv3 tokenizer.
        device: Inference device.
        threshold: Confidence threshold for ``classified`` flag.
        batch_size: Sliding windows per inference batch.
        num_workers: Number of worker processes for FASTA DataLoader.
        max_length: Maximum nucleotide context per sliding window. Tokenizers
            may apply their normal token-space truncation after windowing.
        use_hierarchical_confidence: If ``False`` (default), use flat SF max
            confidence (``sf_conf``) as the threshold metric. If ``True``, use
            order-level confidence (sum of SF probs for the predicted order,
            ``hier_conf``).
        temperature_scaler: Optional :class:`~glare.ood.TemperatureScaler`.
            When provided, logits are divided by ``T`` before computing
            softmax probabilities (so ``sf_conf`` and ``hier_conf`` use
            calibrated probabilities).
        mahal_detector: Optional fitted :class:`~glare.ood.MahalanobisOOD`.
            For Mahalanobis OOD methods, ``mahal_score`` is computed once per
            record from its mean window trunk embedding. Requires the model to
            expose ``forward_with_embeddings()``.
        mahal_threshold: Optional Mahalanobis distance threshold.  When set
            (and ``mahal_detector`` is provided), sequences with
            ``mahal_score > mahal_threshold`` are additionally flagged as
            unknown (``is_unknown = True``) regardless of confidence.
        label_space: Optional label-space name (e.g. ``"repeatmasker_custom"``,
            ``"panteon30"``, ``"legacy45"``) or serialized label metadata loaded
            from a bundle ``labels.json``.  When provided, the correct
            ``idx2superfamily`` mapping and ``sf_to_order`` mapping are used.
            When ``None`` (default), falls back to the module-level PanTEon
            30-SF constants for backward compatibility.
        record_range: Optional half-open source-position range to predict. The
            input is parsed independently, but only records in this range are
            retained and tokenized.
        include_input_index: Retain the private zero-based source-position
            column used to gather process-per-GPU shards. It is omitted by
            default for backward compatibility.
        input_index_offset: Source index represented by record zero in this
            input file. Used by coordinator-created multi-GPU shards.
        ood_method: ``none``/``hier_conf`` preserve confidence thresholding,
            ``energy`` thresholds negated energy, and ``mahalanobis``/``mahal``
            or ``combined`` use the requested Mahalanobis detector.
        timing_metrics: Optional mutable mapping populated with input setup,
            first-batch, inference, record, batch, and final-batch metrics.
        sliding_window_overlap: Fractional overlap for full-record windows.
            Defaults to ``0.5``, matching sliding-window evaluation. Explicit
            ``None`` preserves legacy one-context tokenizer truncation.

    Returns:
        :class:`pandas.DataFrame` with columns:
        ``id``, ``pred_order``, ``pred_superfamily``, ``sf_conf``, ``hier_conf``,
        ``classified``, ``is_unknown``, ``energy_score``, ``temperature``,
        ``mahal_score``.

        ``pred_order`` is derived from ``pred_superfamily`` by splitting on ``"/"``.
        ``energy_score``: higher = more OOD.
        ``mahal_score``: higher = more OOD (NaN if no detector provided).
        ``temperature``: the temperature used for calibration (1.0 if none).
        ``is_unknown``: ``True`` when the sequence is below the confidence
            threshold or (if ``mahal_threshold`` is set) when
            ``mahal_score > mahal_threshold``.
    """
    setup_started = time.monotonic()
    normalized_ood_method = ood_method.lower()
    if normalized_ood_method == "mahal":
        normalized_ood_method = "mahalanobis"
    supported_ood_methods = {"none", "hier_conf", "energy", "mahalanobis", "combined"}
    if normalized_ood_method not in supported_ood_methods:
        raise ValueError(
            f"Unsupported OOD method {ood_method!r}; expected one of "
            f"{sorted(supported_ood_methods)}."
        )
    needs_mahalanobis = normalized_ood_method in {"mahalanobis", "combined"}
    if needs_mahalanobis and mahal_detector is None:
        raise ValueError(f"OOD method {ood_method!r} requires a Mahalanobis detector.")
    if not needs_mahalanobis:
        mahal_detector = None

    # Lazy import to avoid circular dependency at module level
    from .ood import EnergyScorer

    # ── Resolve label-space-specific mappings ────────────────────────────────
    _idx2sf, _sf_to_order_dict, _order_sf_indices = _label_space_mappings(label_space)
    _sf_to_order_fn = lambda sf: _sf_to_order_dict[sf]

    suffix = fasta_path.suffix.lower()
    if suffix == ".ref":
        raw_records = _load_ref_records(
            fasta_path,
            record_range=record_range,
            input_index_offset=input_index_offset,
        )
    else:
        # Assume FASTA format (.fa, .fasta, .fna, etc.)
        raw_records = _load_fasta_records(
            fasta_path,
            record_range=record_range,
            input_index_offset=input_index_offset,
        )

    if not raw_records:
        return empty_prediction_frame(include_input_index=include_input_index)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataset = SlidingWindowInferenceDataset(
        raw_records,
        tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
        overlap=sliding_window_overlap,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if timing_metrics is not None:
        timing_metrics["input_setup_seconds"] = time.monotonic() - setup_started
        timing_metrics["record_count"] = len(raw_records)

    model.eval()
    model.to(device)

    # Determine temperature
    temperature = temperature_scaler.temperature if temperature_scaler is not None else 1.0

    if needs_mahalanobis and not hasattr(model, "forward_with_embeddings"):
        raise ValueError(
            "Mahalanobis prediction requires model.forward_with_embeddings() "
            "so logits and embeddings can be produced in one backbone pass."
        )

    accumulator = MeanLogitAccumulator(len(raw_records), device="cpu")
    inference_started = time.monotonic()
    first_batch_seconds: float | None = None
    batch_count = 0
    final_batch_size = 0

    for batch in loader:
        batch_count += 1
        seq_idx = batch["seq_idx"]
        final_batch_size = len(seq_idx)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            if needs_mahalanobis:
                _order_logits, sf_logits, trunk_emb = model.forward_with_embeddings(
                    input_ids, attention_mask
                )
            else:
                _order_logits, sf_logits = model(input_ids, attention_mask)
                trunk_emb = None
        if first_batch_seconds is None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            first_batch_seconds = time.monotonic() - inference_started

        accumulator.add(seq_idx, _order_logits, sf_logits, trunk_emb)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing_metrics is not None:
        timing_metrics["first_batch_seconds"] = first_batch_seconds
        timing_metrics["inference_seconds"] = time.monotonic() - inference_started
        timing_metrics["batch_count"] = batch_count
        timing_metrics["final_batch_size"] = final_batch_size

    # Every operation below is record-level: aggregate raw float32 logits once,
    # then calibrate/softmax/derive labels and OOD scores.
    _order_logits_mean, sf_logits_mean = accumulator.means()
    sf_logits_for_probs = (
        temperature_scaler.calibrate(sf_logits_mean)
        if temperature_scaler is not None
        else sf_logits_mean
    )
    sf_probs = torch.softmax(sf_logits_for_probs, dim=-1).numpy()
    sf_conf = sf_probs.max(axis=1)
    sf_pred_idx = sf_probs.argmax(axis=1)
    energy_scores = EnergyScorer.score(sf_logits_mean, T=temperature).numpy()
    pred_orders = [_sf_to_order_fn(_idx2sf[int(i)]) for i in sf_pred_idx]
    hier_conf = np.array([
        sf_probs[i, _order_sf_indices[pred_orders[i]]].sum()
        for i in range(len(sf_pred_idx))
    ])

    if needs_mahalanobis:
        mean_embeddings = accumulator.mean_embeddings().numpy()
        mahal_scores = mahal_detector.score_batch(mean_embeddings)  # type: ignore[union-attr]
    else:
        mahal_scores = np.full(len(raw_records), np.nan)

    if normalized_ood_method in {"none", "hier_conf"}:
        conf_for_threshold = _confidence_for_threshold(
            sf_conf,
            hier_conf,
            use_hierarchical_confidence=(
                normalized_ood_method == "hier_conf" or use_hierarchical_confidence
            ),
        )
    elif normalized_ood_method == "energy":
        conf_for_threshold = -energy_scores
    elif normalized_ood_method == "mahalanobis":
        conf_for_threshold = -mahal_scores
    else:
        def _minmax(values: np.ndarray) -> np.ndarray:
            low, high = values.min(), values.max()
            return (values - low) / (high - low + 1e-12)

        conf_for_threshold = np.sqrt(
            _minmax(-energy_scores) * _minmax(-mahal_scores)
        )

    rows: list[dict] = []
    for i, rec in enumerate(raw_records):
        pred_sf = _idx2sf[int(sf_pred_idx[i])]
        rows.append({
            "id": rec["id"],
            "input_index": int(rec["input_index"]),
            "pred_order": pred_orders[i],
            "pred_superfamily": pred_sf,
            "sf_conf": float(sf_conf[i]),
            "hier_conf": float(hier_conf[i]),
            "classified": bool(conf_for_threshold[i] >= threshold),
            "energy_score": float(energy_scores[i]),
            "temperature": temperature,
            "mahal_score": float(mahal_scores[i]),
        })
    df = pd.DataFrame(rows)

    # is_unknown: True when below confidence threshold
    df["is_unknown"] = ~df["classified"]

    # Additionally flag as unknown when mahal_score exceeds mahal_threshold
    if needs_mahalanobis and mahal_threshold is not None:
        df["is_unknown"] = df["is_unknown"] | (df["mahal_score"] > mahal_threshold)

    df = apply_unknown_prediction_labels(df)

    preferred_columns = [
        "input_index",
        "id",
        "pred_order",
        "pred_superfamily",
        "raw_pred_order",
        "raw_pred_superfamily",
        "repeatmasker_class",
        "raw_repeatmasker_class",
        "sf_conf",
        "hier_conf",
        "classified",
        "energy_score",
        "temperature",
        "mahal_score",
        "is_unknown",
    ]
    result = df[[col for col in preferred_columns if col in df.columns]]
    if not include_input_index:
        result = result.drop(columns=["input_index"])
    return result
