"""Data loading and preprocessing for GLARE.

Supports two workflows:
  1. **PanTEon** (new, primary): parse a single FASTA file with PanTEon header
     format and use a 30-superfamily / 8-order vocabulary.
  2. **Repbase** (legacy): parse ``.ref`` files with a TOML mapping to the old
     45-superfamily / 7-order vocabulary.  Kept for backward compatibility.
"""


from .constants import (
    ORDER2IDX,
    IDX2SUPERFAMILY,
    SUPERFAMILY2IDX,
    ORDER_LABELS,
    SUPERFAMILY_LABELS,
    LEGACY_REPBASE_ORDER_LABELS,
    LEGACY_REPBASE_ORDER2IDX,
    LEGACY_REPBASE_SUPERFAMILY_LABELS,
    LEGACY_REPBASE_SUPERFAMILY2IDX,
    get_label_space,
    normalize_sf,
    SF_TO_ORDER,
)
import tomllib


import csv
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - only used in lightweight parse-only envs
    AutoTokenizer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Reverse-complement translation table (module-level, computed once).
RC_MAP = str.maketrans("ACGTacgt", "TGCAtgca")

# ===========================================================================
# PlantCAD2 character-level tokenizer
# ===========================================================================

#: Character → token-ID mapping for PlantCAD2.
_PLANTCAD2_CHAR_MAP: dict[str, int] = {"A": 3, "T": 6, "C": 4, "G": 5, "N": 2}
_PLANTCAD2_PAD_ID: int = 0


class PlantCAD2Tokenizer:
    """Minimal character-level tokenizer for PlantCAD2.

    PlantCAD2 has no HuggingFace tokenizer files on the Hub.  This class
    replicates the expected ``AutoTokenizer`` call interface used by
    :class:`TEDataset` and :class:`SlidingWindowDataset`:

    .. code-block:: python

        encoding = tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=128,
        )
        # encoding["input_ids"]     → torch.LongTensor of shape (1, L)
        # encoding["attention_mask"] → torch.LongTensor of shape (1, L)

    Unknown characters are mapped to the ``N`` token (ID 2).

    Attributes:
        pad_token_id: Token ID used for padding (0).
    """

    pad_token_id: int = _PLANTCAD2_PAD_ID

    def __call__(
        self,
        sequence: str,
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 1024,
        pad_to_multiple_of: int = 128,
        **kwargs,
    ) -> dict:
        """Encode *sequence* into input_ids and attention_mask tensors.

        Args:
            sequence: Raw nucleotide string (A/T/C/G/N).
            return_tensors: Only ``"pt"`` (PyTorch) is supported.
            padding: Only ``"max_length"`` is supported.
            truncation: If ``True``, truncate to *max_length*.
            max_length: Target sequence length after truncation/padding.
            pad_to_multiple_of: Round *max_length* up to the nearest multiple
                of this value (mirrors HuggingFace tokenizer behaviour).
            **kwargs: Ignored (for API compatibility).

        Returns:
            Dict with keys ``"input_ids"`` and ``"attention_mask"``, each a
            ``torch.LongTensor`` of shape ``(1, padded_length)``.
        """
        # Round max_length up to nearest multiple of pad_to_multiple_of
        if pad_to_multiple_of and pad_to_multiple_of > 1:
            remainder = max_length % pad_to_multiple_of
            if remainder != 0:
                max_length = max_length + (pad_to_multiple_of - remainder)

        # Truncate
        if truncation and len(sequence) > max_length:
            sequence = sequence[:max_length]

        # Encode characters → IDs (unknown → N token)
        n_id = _PLANTCAD2_CHAR_MAP["N"]
        ids = [_PLANTCAD2_CHAR_MAP.get(ch, n_id) for ch in sequence]
        mask = [1] * len(ids)

        # Pad to max_length
        pad_len = max_length - len(ids)
        ids = ids + [_PLANTCAD2_PAD_ID] * pad_len
        mask = mask + [0] * pad_len

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.tensor([mask], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def is_dnabert2_model_name(base_model_name: str) -> bool:
    """Return ``True`` for DNABERT-2 HuggingFace model identifiers/paths."""
    normalized = base_model_name.lower().replace("_", "-")
    return "dnabert-2" in normalized or "dnabert2" in normalized


def is_caduceus_model_name(base_model_name: str) -> bool:
    """Return ``True`` for Caduceus HuggingFace model identifiers/paths."""
    return "caduceus" in base_model_name.lower()


class DNABERT2Tokenizer:
    """Wrapper that applies GLARE nucleotide-window semantics to DNABERT-2.

    DNABERT-2 uses BPE tokenization, so GLARE's configured ``max_length`` is
    treated as a nucleotide context length first.  The raw DNA string is
    uppercased and cropped at the nucleotide level before delegating to the
    underlying HuggingFace tokenizer, which then performs normal BPE
    truncation/padding in token space.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)

    def __call__(self, sequence: str, *args, **kwargs) -> dict:
        max_length = kwargs.get("max_length")
        sequence = sequence.upper()
        if max_length is not None and len(sequence) > int(max_length):
            sequence = sequence[: int(max_length)]
        return self.tokenizer(sequence, *args, **kwargs)


class CaduceusTokenizer:
    """Wrapper that preserves GLARE nucleotide-window semantics for Caduceus.

    The Caduceus HuggingFace tokenizer is character/nucleotide-level for DNA:
    ``A/C/G/T/N`` each map to one token when special tokens are disabled. GLARE
    therefore treats the configured ``max_length`` as a nucleotide context
    length, uppercases and crops raw DNA before tokenization, and disables
    default special-token insertion so one nucleotide remains one token.

    If a future Caduceus tokenizer variant is not one-token-per-nucleotide, the
    wrapper still performs nucleotide-level cropping before delegating to the
    underlying tokenizer's normal token-space truncation/padding.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.is_nucleotide_level = self._detect_nucleotide_level()

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)

    def _detect_nucleotide_level(self) -> bool:
        """Best-effort check that ``A/C/G/T/N`` encode as one token each."""
        try:
            encoded = self.tokenizer("ACGTN", add_special_tokens=False)
            input_ids = encoded.get("input_ids", [])
            return len(input_ids) == 5
        except Exception:
            return False

    def __call__(self, sequence: str, *args, **kwargs) -> dict:
        max_length = kwargs.get("max_length")
        sequence = sequence.upper()
        if max_length is not None and len(sequence) > int(max_length):
            sequence = sequence[: int(max_length)]

        # Caduceus' default tokenizer appends a [SEP] token. GLARE windows are
        # nucleotide contexts, so omit special tokens for one-token-per-base
        # Caduceus tokenizers unless a caller explicitly requests otherwise.
        if self.is_nucleotide_level and "add_special_tokens" not in kwargs:
            kwargs["add_special_tokens"] = False

        return self.tokenizer(sequence, *args, **kwargs)


def get_tokenizer(base_model_name: str):
    """Return the appropriate tokenizer for *base_model_name*.

    * **PlantCAD2** (name contains ``"plantcad"``, case-insensitive): returns
      a :class:`PlantCAD2Tokenizer` instance because PlantCAD2 has no tokenizer
      files on the Hub.
    * **Caduceus** (name contains ``"caduceus"``, case-insensitive): wraps the
      HuggingFace tokenizer in :class:`CaduceusTokenizer` so ``max_length`` is a
      nucleotide context length and DNA is uppercased before tokenization.
    * **DNABERT-2** (name contains ``"dnabert-2"`` or ``"dnabert2"``): returns
      a :class:`DNABERT2Tokenizer` wrapper around the HuggingFace tokenizer so
      ``max_length`` remains a nucleotide context length before BPE tokenization.
    * **All other models**: delegates to
      ``AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)``.

    Args:
        base_model_name: HuggingFace model identifier or local path.

    Returns:
        A tokenizer instance compatible with the :class:`TEDataset` interface.
    """
    name_lower = base_model_name.lower()
    if "plantcad" in name_lower:
        return PlantCAD2Tokenizer()
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required to load HuggingFace tokenizers; install it "
            "or run inside the GLARE evaluation environment."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True)
    if is_caduceus_model_name(base_model_name):
        return CaduceusTokenizer(tokenizer)
    if is_dnabert2_model_name(base_model_name):
        return DNABERT2Tokenizer(tokenizer)
    return tokenizer


# RepeatMasker-style labels that are intentionally not collapsed into the
# PanTEon/GLARE 30-label vocabulary.  These are broad, structural, unsupported,
# or otherwise ambiguous labels; they are skipped and reported in diagnostics.
AMBIGUOUS_REPEATMASKER_LABELS: set[str] = {
    "",
    "Unknown",
    "Other",
    "DNA",
    "LTR",
    "LINE",
    "Satellite",
    "Structural_RNA",
    "LTR/Caulimovirus",
}


# Explicit, conservative mapping from RepeatMasker-style labels produced by
# glare/data/repbase-to-repeatmasker.toml to the 30 PanTEon/GLARE
# superfamily tokens used by Poales-holdout models.  Ambiguous labels such as
# "DNA", "LTR", "LINE", "Satellite", "LTR/Caulimovirus", "LINE/L2", and structural RNAs are
# intentionally omitted and therefore reported as unmapped rather than forced
# into the 30-label space.
REPEATMASKER_TO_PANTEON_SF: dict[str, str] = {
    "DNA/Academ": "ACADEM-1",
    "LTR/Pao": "BELPAO",
    "DNA/CMC": "CACTA",           # CMC/EnSpm/CACTA → PanTEon CACTA
    "LTR/Copia": "COPIA",
    "LINE/CR1": "CR1",
    "DNA/Crypton": "CRYPTON",
    "LTR/DIRS": "DIRS",
    "LTR/ERV": "ERV",
    "LTR/Gypsy": "GYPSY",
    "DNA/hAT": "HAT",
    "RC/Helitron": "HELITRON",
    "LINE/I": "I",
    "DNA/Kolobok": "KOLOBOK",
    "LINE/L1": "L1",
    "DNA/Merlin": "MERLIN",
    "DNA/MULE": "MULE",
    "DNA/P": "P",
    "PLE": "PLE",
    "DNA/Harbinger": "PIFHARBINGER",
    "DNA/PiggyBac": "PIGGYBAC",
    "LINE/R1": "R1",
    "LINE/RTE": "RTE",
    "SINE": "SINE",
    "DNA/TcMar": "TC1MARINER",
    "SINE/tRNA": "TRNA",
}


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    """Return a stable, alphabetically sorted plain dict from a Counter."""
    return dict(sorted(counter.items()))


def count_record_labels(records: list[dict]) -> dict[str, dict[str, int]]:
    """Count record superfamily and order labels for logging/diagnostics."""
    return {
        "superfamily_counts": _counter_dict(
            Counter(str(r.get("superfamily", "")) for r in records)
        ),
        "order_counts": _counter_dict(
            Counter(str(r.get("order", "")) for r in records)
        ),
    }


def format_label_distribution(records: list[dict], title: str) -> str:
    """Format superfamily/order distributions for readable logs."""
    counts = count_record_labels(records)
    lines = [f"{title} (n={len(records)})", "orders:"]
    lines.extend(
        f"  {label}: {count}"
        for label, count in counts["order_counts"].items()
    )
    lines.append("superfamilies:")
    lines.extend(
        f"  {label}: {count}"
        for label, count in counts["superfamily_counts"].items()
    )
    return "\n".join(lines)


def _ordered_superfamily_labels(labels: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    """Return labels in PanTEon vocabulary order, then any unknowns alphabetically."""
    label_set = {str(label) for label in labels}
    ordered = [label for label in SUPERFAMILY_LABELS if label in label_set]
    ordered.extend(sorted(label_set - set(ordered)))
    return ordered


def _superfamily_order(label: str) -> str:
    """Return the coarse order for PanTEon tokens or compound labels."""
    if label in SF_TO_ORDER:
        return SF_TO_ORDER[label]
    if "/" in label:
        return label.split("/", 1)[0]
    return label


def _label_space_name(data_config: Optional[dict[str, Any]]) -> str:
    """Resolve the configured label-space name from a data config."""
    data_config = data_config or {}
    return str(
        data_config.get("label_space")
        or data_config.get("repbase_label_space")
        or "panteon30"
    )


def _is_custom_repeatmasker_label_space(label_space_name: str) -> bool:
    """Return whether *label_space_name* selects the custom RepeatMasker space."""
    return label_space_name.strip().lower().replace("-", "_") in {
        "repeatmasker_custom",
        "custom_repeatmasker",
        "rm_custom",
        "repeatmasker32",
        "32",
    }


def filter_records_by_superfamilies(
    records: list[dict],
    labels: set[str] | list[str] | tuple[str, ...],
    dataset_name: str = "records",
    log: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Filter records to a conservative superfamily label subset.

    Args:
        records: Record dicts containing a ``superfamily`` key.
        labels: Allowed superfamily labels.
        dataset_name: Human-readable name used in diagnostics/logging.
        log: If ``True``, emit kept/skipped counts via the module logger.

    Returns:
        ``(filtered_records, stats)`` where ``stats`` includes total, kept,
        skipped, kept/skipped per-label counts, and the allowed label list.
    """
    allowed = set(str(label) for label in labels)
    allowed_ordered = _ordered_superfamily_labels(allowed)
    kept: list[dict] = []
    kept_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for rec in records:
        sf = str(rec.get("superfamily", ""))
        if sf in allowed:
            kept.append(rec)
            kept_counts[sf] += 1
        else:
            skipped_counts[sf] += 1

    stats: dict[str, Any] = {
        "dataset_name": dataset_name,
        "allowed_labels": allowed_ordered,
        "allowed_label_count": len(allowed_ordered),
        "total_records": len(records),
        "kept_records": len(kept),
        "skipped_records": len(records) - len(kept),
        "kept_superfamily_counts": _counter_dict(kept_counts),
        "skipped_superfamily_counts": _counter_dict(skipped_counts),
    }
    if log:
        logger.info(
            "Label subset filter [%s]: kept %d/%d records; skipped %d; labels=%s",
            dataset_name,
            stats["kept_records"],
            stats["total_records"],
            stats["skipped_records"],
            ",".join(allowed_ordered),
        )
        if skipped_counts:
            logger.info(
                "Label subset filter [%s] skipped superfamily counts: %s",
                dataset_name,
                stats["skipped_superfamily_counts"],
            )
    return kept, stats


def filter_records_by_superfamily_min_count(
    records: list[dict],
    min_count: int | None,
    *,
    dataset_name: str = "records",
    log: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Keep only superfamilies represented by at least ``min_count`` records."""
    if min_count is None or int(min_count) <= 1:
        counts = Counter(str(rec.get("superfamily", "")) for rec in records)
        return list(records), {
            "dataset_name": dataset_name,
            "enabled": False,
            "min_count": min_count,
            "total_records": len(records),
            "kept_records": len(records),
            "skipped_records": 0,
            "kept_superfamilies": _ordered_superfamily_labels(set(counts)),
            "kept_superfamily_count": len(counts),
            "kept_superfamily_counts": _counter_dict(counts),
            "skipped_superfamily_counts": {},
        }

    threshold = int(min_count)
    counts = Counter(str(rec.get("superfamily", "")) for rec in records)
    allowed = {sf for sf, count in counts.items() if count >= threshold}
    kept = [rec for rec in records if str(
        rec.get("superfamily", "")) in allowed]
    kept_counts = Counter(str(rec.get("superfamily", "")) for rec in kept)
    skipped_counts = Counter(
        str(rec.get("superfamily", ""))
        for rec in records
        if str(rec.get("superfamily", "")) not in allowed
    )
    stats: dict[str, Any] = {
        "dataset_name": dataset_name,
        "enabled": True,
        "min_count": threshold,
        "total_records": len(records),
        "kept_records": len(kept),
        "skipped_records": len(records) - len(kept),
        "kept_superfamilies": _ordered_superfamily_labels(allowed),
        "kept_superfamily_count": len(allowed),
        "kept_superfamily_counts": _counter_dict(kept_counts),
        "skipped_superfamily_counts": _counter_dict(skipped_counts),
    }
    if log:
        logger.info(
            "Superfamily min-count filter [%s]: min_count=%d kept %d/%d records; "
            "kept_superfamilies=%d; skipped=%s",
            dataset_name,
            threshold,
            stats["kept_records"],
            stats["total_records"],
            stats["kept_superfamily_count"],
            stats["skipped_superfamily_counts"],
        )
    return kept, stats


def label_subset_enabled(label_subset_config: Optional[dict[str, Any]]) -> bool:
    """Return ``True`` when a data.label_subset config is active."""
    if not label_subset_config:
        return False
    return bool(label_subset_config.get("enabled", False))


def _resolve_label_subset_path(
    label_subset_config: dict[str, Any],
    data_config: dict[str, Any],
    primary_key: str,
    fallback_data_key: str | None,
    default: str,
) -> Path:
    """Resolve a path from label_subset, data config, or a conservative default."""
    value = label_subset_config.get(primary_key)
    if value is None and fallback_data_key is not None:
        value = data_config.get(fallback_data_key)
    if value is None:
        value = default
    return Path(str(value))


def _label_subset_taxonomy_holdout(
    label_subset_config: dict[str, Any],
    data_config: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Return the taxonomy holdout config used for PanTEon train-side labels."""
    return (
        label_subset_config.get("panteon_taxonomy_holdout")
        or label_subset_config.get("taxonomy_holdout")
        or data_config.get("taxonomy_holdout")
    )


def _taxonomy_holdout_label(config: Optional[dict[str, Any]]) -> str:
    """Return a compact label for a taxonomy holdout configuration."""
    if not config:
        return "taxonomy_holdout"
    raw_name = config.get("name") or config.get(
        "test_name") or config.get("filter_value")
    if raw_name is None and "filter_values" in config:
        raw_values = config.get("filter_values")
        if isinstance(raw_values, (list, tuple, set)):
            raw_name = "_".join(str(v) for v in raw_values)
        else:
            raw_name = raw_values
    label = str(raw_name or "taxonomy_holdout").strip().lower()
    label = "".join(ch if ch.isalnum() else "_" for ch in label)
    label = "_".join(part for part in label.split("_") if part)
    return label or "taxonomy_holdout"


def _select_panteon_records_by_taxonomy(
    records: list[dict],
    taxonomy_config: dict[str, Any],
    *,
    context: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Select PanTEon records whose FASTA species match a metadata taxonomy filter."""
    selected_species, metadata_stats = load_taxonomy_holdout_species(
        taxonomy_config)
    species_match = str(taxonomy_config.get(
        "species_match", "normalized_exact"))
    fasta_species_mode = str(taxonomy_config.get(
        "fasta_species_mode", "after_at"))
    fail_on_no_matches = bool(taxonomy_config.get("fail_on_no_matches", True))

    if not selected_species:
        msg = (
            f"{context} selected zero species from metadata "
            f"{metadata_stats['metadata_csv']} using "
            f"{metadata_stats['filter_column']}={metadata_stats['filter_values']}."
        )
        if fail_on_no_matches:
            raise ValueError(msg)
        logger.warning(msg)

    matches: list[dict] = []
    matched_fasta_species: set[str] = set()
    n_missing_fasta_species = 0
    for rec in records:
        if fasta_species_mode == "after_at":
            fasta_species = rec.get("species") or extract_panteon_header_species(
                rec.get("id", ""), mode=fasta_species_mode
            )
        else:
            fasta_species = extract_panteon_header_species(
                rec.get("id", ""), mode=fasta_species_mode
            )
        if not fasta_species:
            n_missing_fasta_species += 1
        if _species_matches_holdout(fasta_species, selected_species, species_match):
            matches.append(rec)
            matched_fasta_species.add(normalize_species_name(fasta_species))

    if not matches:
        msg = (
            f"{context} matched zero FASTA records. Check metadata species names, "
            "FASTA header species extraction, and species_match settings."
        )
        if fail_on_no_matches:
            raise ValueError(msg)
        logger.warning(msg)

    stats = {
        **metadata_stats,
        "species_match": species_match,
        "fasta_species_mode": fasta_species_mode,
        "total_records": len(records),
        "matched_records": len(matches),
        "matched_fasta_species": len(matched_fasta_species),
        "unmatched_metadata_species": max(
            len(selected_species - matched_fasta_species), 0
        ),
        "missing_fasta_species_records": n_missing_fasta_species,
    }
    return matches, stats


def filter_panteon_records_by_taxonomy(
    records: list[dict],
    taxonomy_filter: Optional[dict[str, Any]] = None,
    *,
    filter_name: str = "taxonomy_filter",
) -> tuple[list[dict], dict[str, Any]]:
    """Filter PanTEon records by metadata taxonomy without creating a holdout.

    This is used for same-clade training/evaluation experiments, e.g. train only
    on records whose metadata ``Kingdom`` is ``Plantae`` and then make the usual
    random train/validation/test split within that selected subset.  Existing
    configs are unaffected because filtering is disabled unless a metadata CSV is
    provided in the new filter config.
    """
    if not taxonomy_holdout_enabled(taxonomy_filter):
        return list(records), {
            "taxonomy_filter_enabled": False,
            "filter_name": filter_name,
            "total_records": len(records),
            "matched_records": len(records),
        }

    assert taxonomy_filter is not None
    matches, stats = _select_panteon_records_by_taxonomy(
        records,
        taxonomy_filter,
        context=filter_name,
    )
    return matches, {
        **stats,
        "taxonomy_filter_enabled": True,
        "filter_name": filter_name,
    }


def build_taxonomy_group_test_sets(
    records: list[dict],
    taxonomy_groups: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Build named PanTEon test sets from metadata-driven taxonomy groups.

    Unlike :func:`partition_panteon_records_by_taxonomy_holdout`, this helper is
    evaluation-only: it filters the already-created held-out test split into
    named groups and does not alter training or validation records.
    """
    if isinstance(taxonomy_groups, dict):
        group_configs = []
        for group_name, group_cfg in taxonomy_groups.items():
            cfg = dict(group_cfg or {})
            cfg.setdefault("name", str(group_name))
            group_configs.append(cfg)
    else:
        group_configs = [dict(group_cfg or {})
                         for group_cfg in taxonomy_groups]

    grouped_records: dict[str, list[dict]] = {}
    group_stats: dict[str, Any] = {
        "total_records": len(records),
        "groups": {},
    }

    for group_cfg in group_configs:
        if group_cfg.get("enabled", True) is False:
            continue

        group_name = _taxonomy_holdout_label(group_cfg)
        if not group_name.startswith("panteon_"):
            test_name = f"panteon_{group_name}"
        else:
            test_name = group_name

        matches, match_stats = _select_panteon_records_by_taxonomy(
            records,
            group_cfg,
            context=f"taxonomy test group {test_name!r}",
        )

        grouped_records[test_name] = matches
        group_stats["groups"][test_name] = {
            **match_stats,
            "test_name": test_name,
        }

    return grouped_records, group_stats


def partition_panteon_records_by_taxonomy_holdout(
    records: list[dict],
    taxonomy_holdout: Optional[dict[str, Any]] = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Partition PanTEon records into train-side and taxonomy-holdout records.

    The train-side records are the non-holdout records before the random
    train/validation split.  This is the conservative source used when computing
    Repbase↔PanTEon shared training labels.
    """
    if not taxonomy_holdout_enabled(taxonomy_holdout):
        stats = {
            "taxonomy_holdout_enabled": False,
            "split_mode": "no_taxonomy_holdout_all_records_train_side",
            "total_records": len(records),
            "train_side_records": len(records),
            "holdout_records": 0,
        }
        return list(records), [], stats

    assert taxonomy_holdout is not None
    holdout_recs, filter_stats = _select_panteon_records_by_taxonomy(
        records,
        taxonomy_holdout,
        context="taxonomy_holdout",
    )
    holdout_ids = {id(rec) for rec in holdout_recs}
    train_side_recs = [rec for rec in records if id(rec) not in holdout_ids]

    if not train_side_recs:
        raise ValueError(
            "taxonomy_holdout assigned all records to test; no records remain for "
            "training/validation."
        )

    stats = {
        **filter_stats,
        "taxonomy_holdout_enabled": True,
        "split_mode": "taxonomy_holdout_partition",
        "train_side_records": len(train_side_recs),
        "holdout_records": len(holdout_recs),
    }
    return train_side_recs, holdout_recs, stats


def compute_train_intersection_label_subset(
    label_subset_config: dict[str, Any],
    data_config: Optional[dict[str, Any]] = None,
) -> tuple[list[str], dict[str, Any]]:
    """Compute labels present in both Repbase and PanTEon training sources.

    Repbase labels are taken from mapped records under the configured Repbase
    train directory.  PanTEon labels are taken from the non-holdout train side
    after applying the configured taxonomy holdout (e.g. non-Poales records when
    ``Order == Poales`` is held out).  Test/holdout records are intentionally not
    used to define the intersection.
    """
    data_config = data_config or {}
    repbase_train_dir = _resolve_label_subset_path(
        label_subset_config,
        data_config,
        "repbase_train_dir",
        "repbase_dir",
        "glare/data/repbase31_filtered",
    )
    repbase_toml_path = _resolve_label_subset_path(
        label_subset_config,
        data_config,
        "repbase_toml_path",
        "toml_mapping",
        "glare/data/repbase-to-repeatmasker.toml",
    )
    panteon_fasta_path = _resolve_label_subset_path(
        label_subset_config,
        data_config,
        "panteon_fasta_path",
        "fasta_path",
        "glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta",
    )
    taxonomy_holdout = _label_subset_taxonomy_holdout(
        label_subset_config, data_config)

    label_space_name = _label_space_name(data_config)
    panteon_label_mapping = (
        label_subset_config.get("panteon_label_mapping")
        or data_config.get("panteon_label_mapping")
    )
    if _is_custom_repeatmasker_label_space(label_space_name):
        repbase_records, repbase_diagnostics = load_repbase_dir_for_label_space(
            repbase_train_dir,
            repbase_toml_path,
            label_space_name=label_space_name,
        )
        panteon_records = load_panteon_dataset(
            panteon_fasta_path,
            label_mapping_path=Path(
                str(panteon_label_mapping)) if panteon_label_mapping else None,
            label_space_name=label_space_name,
        )
    else:
        repbase_records, repbase_diagnostics = load_repbase_dir_for_panteon_training(
            repbase_train_dir,
            repbase_toml_path,
        )
        panteon_records = load_panteon_dataset(panteon_fasta_path)
    panteon_train_side_records, panteon_holdout_records, partition_stats = (
        partition_panteon_records_by_taxonomy_holdout(
            panteon_records,
            taxonomy_holdout=taxonomy_holdout,
        )
    )

    repbase_counts = Counter(str(r.get("superfamily", ""))
                             for r in repbase_records)
    panteon_counts = Counter(
        str(r.get("superfamily", "")) for r in panteon_train_side_records
    )
    repbase_labels = set(repbase_counts)
    panteon_labels = set(panteon_counts)
    intersection = _ordered_superfamily_labels(repbase_labels & panteon_labels)

    if not intersection:
        raise ValueError(
            "data.label_subset train_intersection produced zero shared labels between "
            f"Repbase train source {repbase_train_dir} and PanTEon train-side source "
            f"{panteon_fasta_path}."
        )

    stats: dict[str, Any] = {
        "mode": "train_intersection",
        "repbase_train_dir": str(repbase_train_dir),
        "repbase_toml_path": str(repbase_toml_path),
        "panteon_fasta_path": str(panteon_fasta_path),
        "panteon_label_mapping": str(panteon_label_mapping) if panteon_label_mapping else None,
        "label_space": label_space_name,
        "taxonomy_holdout": taxonomy_holdout,
        "repbase_loaded_records": repbase_diagnostics["loaded_records"],
        "repbase_mapped_records": len(repbase_records),
        "repbase_skipped_records": repbase_diagnostics["skipped_records"],
        "panteon_total_records": len(panteon_records),
        "panteon_train_side_records": len(panteon_train_side_records),
        "panteon_holdout_records": len(panteon_holdout_records),
        "panteon_partition_stats": partition_stats,
        "repbase_train_superfamily_counts": _counter_dict(repbase_counts),
        "panteon_train_side_superfamily_counts": _counter_dict(panteon_counts),
        "intersection_labels": intersection,
        "intersection_label_count": len(intersection),
        "repbase_only_labels_excluded": _ordered_superfamily_labels(repbase_labels - panteon_labels),
        "panteon_only_labels_excluded": _ordered_superfamily_labels(panteon_labels - repbase_labels),
        "vocab_labels_absent_from_repbase_train": _ordered_superfamily_labels(
            set(get_label_space(label_space_name)[
                "superfamily_labels"]) - repbase_labels
        ),
        "vocab_labels_absent_from_panteon_train_side": _ordered_superfamily_labels(
            set(get_label_space(label_space_name)[
                "superfamily_labels"]) - panteon_labels
        ),
    }
    return intersection, stats


def resolve_superfamily_label_subset(
    data_config: Optional[dict[str, Any]],
) -> tuple[list[str] | None, dict[str, Any] | None]:
    """Resolve ``data.label_subset`` into an allowed superfamily list.

    Default behavior is preserved: if ``data.label_subset.enabled`` is unset or
    false, returns ``(None, None)`` and callers should skip filtering.
    """
    data_config = data_config or {}
    label_subset_config = data_config.get("label_subset")
    if not label_subset_enabled(label_subset_config):
        return None, None

    assert label_subset_config is not None
    mode = str(label_subset_config.get("mode", "explicit")).strip().lower()
    if mode == "explicit":
        raw_labels = label_subset_config.get("labels") or []
        if not isinstance(raw_labels, (list, tuple, set)):
            raise ValueError(
                "data.label_subset.labels must be a list in explicit mode.")
        labels = [str(label).strip()
                  for label in raw_labels if str(label).strip()]
        if not labels:
            raise ValueError(
                "data.label_subset explicit mode requires at least one label.")
        label_space = get_label_space(_label_space_name(data_config))
        unknown = sorted(set(labels) - set(label_space["superfamily_labels"]))
        if unknown:
            raise ValueError(
                "data.label_subset.labels contains labels outside the configured "
                f"{label_space['name']} vocabulary: {unknown}"
            )
        labels_ordered = _ordered_superfamily_labels(labels)
        stats = {
            "mode": "explicit",
            "intersection_labels": labels_ordered,
            "intersection_label_count": len(labels_ordered),
        }
        return labels_ordered, stats

    if mode in {"train_intersection", "intersection"}:
        return compute_train_intersection_label_subset(label_subset_config, data_config)

    raise ValueError(
        f"Unsupported data.label_subset.mode {mode!r}; expected 'explicit' or "
        "'train_intersection'."
    )


# ===========================================================================
# PanTEon FASTA parsing
# ===========================================================================

def normalize_species_name(species: str) -> str:
    """Normalize a species name for robust metadata ↔ FASTA matching.

    The PanTEon FASTA embeds species names after an ``@`` in the header, while
    the metadata CSV stores species in a plain text column.  This normalisation
    removes header-specific punctuation, treats underscores and punctuation as
    spaces, collapses repeated whitespace, and case-folds for matching.
    """
    cleaned = species.strip().lstrip("@").replace("_", " ")
    cleaned = "".join(ch if ch.isalnum() or ch.isspace()
                      else " " for ch in cleaned)
    return " ".join(cleaned.split()).casefold()


def extract_panteon_header_species(
    header: str,
    mode: str = "after_at",
) -> str:
    """Extract the species name embedded in a PanTEon FASTA header.

    Args:
        header: FASTA header without the leading ``>``.
        mode: Extraction strategy.  Supported values are:
            ``"after_at"`` (default): text after the first ``@``;
            ``"after_first_space"``: text after the first whitespace token;
            ``"full_header"``: the whole header.

    Returns:
        Extracted species string, or ``""`` if no species-like text is found.
    """
    header = header.strip()
    if mode == "after_at":
        if "@" not in header:
            return ""
        return header.split("@", 1)[1].strip()
    if mode == "after_first_space":
        parts = header.split(maxsplit=1)
        if len(parts) < 2:
            return ""
        return parts[1].strip().lstrip("@").strip()
    if mode == "full_header":
        return header
    raise ValueError(
        f"Unsupported FASTA species extraction mode {mode!r}; expected one of "
        "'after_at', 'after_first_space', or 'full_header'."
    )


def parse_panteon_fasta(fasta_path: Path) -> list[dict]:
    """Parse a PanTEon FASTA file into a list of record dicts.

    Header format::

        >ElementName#ClassI/Order/Superfamily  Species name

    Label extraction::

        raw_sf = header.split(" ")[0].split("#")[1].split("/")[-1]
        sf     = normalize_sf(raw_sf)

    Each returned record has keys:
        ``id``, ``sequence``, ``superfamily``, ``order``, ``species``.

    Records whose normalized superfamily is not in :data:`SUPERFAMILY_LABELS`
    are **not** skipped here — use :func:`load_panteon_dataset` for filtered
    loading.

    Args:
        fasta_path: Path to the PanTEon FASTA file.

    Returns:
        List of record dicts (unfiltered).
    """
    records: list[dict] = []
    current_id: Optional[str] = None
    current_sf: Optional[str] = None
    current_order: Optional[str] = None
    current_species: Optional[str] = None
    current_native_label: Optional[str] = None
    seq_parts: list[str] = []

    with open(fasta_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Save previous record
                if current_id is not None:
                    records.append({
                        "id": current_id,
                        "sequence": "".join(seq_parts).upper(),
                        "superfamily": current_sf,
                        "order": current_order,
                        "native_label": current_native_label,
                        "species": current_species or "",
                    })

                # Parse new header
                header = line[1:]  # strip '>'
                # id token is everything before the first space
                id_token = header.split(" ")[0]
                # Extract classification path after '#'
                if "#" in id_token:
                    class_path = id_token.split("#")[1]
                    class_parts = class_path.split("/")
                    raw_order = class_parts[-2] if len(
                        class_parts) >= 2 else ""
                    raw_sf = class_parts[-1]
                else:
                    raw_order = ""
                    raw_sf = ""

                sf = normalize_sf(raw_sf) if raw_sf else ""
                order = SF_TO_ORDER.get(sf, "") if sf in SF_TO_ORDER else ""
                native_label = f"{raw_order.strip().upper()}/{sf}" if raw_order and sf else ""

                current_id = header  # full header line (without '>')
                current_sf = sf
                current_order = order
                current_native_label = native_label
                current_species = extract_panteon_header_species(header)
                seq_parts = []
            else:
                stripped = line.strip()
                if stripped and current_id is not None:
                    seq_parts.append(stripped)

    # Save last record
    if current_id is not None:
        records.append({
            "id": current_id,
            "sequence": "".join(seq_parts).upper(),
            "superfamily": current_sf,
            "order": current_order,
            "native_label": current_native_label,
            "species": current_species or "",
        })

    return records


def load_panteon_dataset(
    fasta_path: Path,
    label_mapping_path: Optional[Path] = None,
    label_space_name: str = "panteon30",
) -> list[dict]:
    """Load and filter the PanTEon FASTA file.

    Parses the FASTA and drops records whose superfamily is not in
    :data:`SUPERFAMILY_LABELS` (i.e. unknown / unmappable labels).

    Args:
        fasta_path: Path to the PanTEon FASTA file.

    Returns:
        Filtered list of record dicts, each with keys
        ``id``, ``sequence``, ``superfamily``, ``order``.
    """
    all_records = parse_panteon_fasta(fasta_path)
    if label_mapping_path is not None:
        with open(label_mapping_path, "rb") as fh:
            mapping: dict[str, str] = tomllib.load(fh)
        label_space = get_label_space(label_space_name)
        sf_set = set(label_space["superfamily_labels"])
        sf_to_order_map = label_space["sf_to_order"]
        filtered = []
        for rec in all_records:
            source_label = str(rec.get("native_label") or "")
            mapped_label = mapping.get(source_label)
            if mapped_label is None or mapped_label not in sf_set:
                continue
            mapped = dict(rec)
            mapped["native_order"] = rec.get("order", "")
            mapped["native_superfamily"] = rec.get("superfamily", "")
            mapped["repeatmasker_label"] = mapped_label
            mapped["superfamily"] = mapped_label
            mapped["order"] = str(sf_to_order_map.get(
                mapped_label, _superfamily_order(mapped_label)))
            filtered.append(mapped)
        return filtered

    sf_set = set(SUPERFAMILY_LABELS)
    filtered = [r for r in all_records if r["superfamily"] in sf_set]
    return filtered


def _clean_metadata_column_name(name: str) -> str:
    """Strip whitespace and UTF-8 BOM artifacts from CSV column names."""
    return name.strip().lstrip("\ufeff")


def _metadata_column_lookup(fieldnames: list[str] | None) -> dict[str, str]:
    """Map cleaned CSV column names to their original DictReader keys."""
    if not fieldnames:
        return {}
    return {_clean_metadata_column_name(name): name for name in fieldnames}


def _matches_metadata_filter(
    observed: str,
    expected_values: list[str],
    match_mode: str,
) -> bool:
    """Return whether a metadata cell satisfies the configured filter."""
    observed_s = observed.strip()
    if match_mode == "exact":
        return observed_s in expected_values
    if match_mode == "casefold":
        observed_cf = observed_s.casefold()
        return observed_cf in {v.casefold() for v in expected_values}
    if match_mode == "contains":
        return any(v in observed_s for v in expected_values)
    if match_mode == "casefold_contains":
        observed_cf = observed_s.casefold()
        return any(v.casefold() in observed_cf for v in expected_values)
    raise ValueError(
        f"Unsupported taxonomy metadata filter match_mode {match_mode!r}; expected "
        "'exact', 'casefold', 'contains', or 'casefold_contains'."
    )


def _metadata_filter_values(config: dict[str, Any]) -> list[str]:
    """Return configured metadata filter value(s) as a list of strings."""
    if "filter_values" in config:
        raw_values = config["filter_values"]
    else:
        raw_values = config.get("filter_value")

    if raw_values is None:
        raise ValueError(
            "taxonomy_holdout requires 'filter_value' or 'filter_values'."
        )
    if isinstance(raw_values, (list, tuple, set)):
        values = [str(v).strip() for v in raw_values]
    else:
        values = [str(raw_values).strip()]
    values = [v for v in values if v]
    if not values:
        raise ValueError("taxonomy_holdout filter values cannot be empty.")
    return values


def taxonomy_holdout_enabled(config: Optional[dict[str, Any]]) -> bool:
    """Return ``True`` when a taxonomy holdout config is active."""
    if not config:
        return False
    if config.get("enabled", True) is False:
        return False
    return bool(config.get("metadata_csv"))


def load_taxonomy_holdout_species(
    config: dict[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    """Load species names selected by a metadata-driven taxonomy holdout.

    Required config keys are ``metadata_csv``, ``filter_column``,
    ``filter_value``/``filter_values``, and ``species_column``.  The metadata
    filter is generic: e.g. ``Order == Poales`` today, but any CSV column/value
    can be used in future experiments.
    """
    metadata_csv = Path(config["metadata_csv"])
    filter_column = str(config.get("filter_column", "")).strip()
    species_column = str(config.get("species_column", "")).strip()
    filter_values = _metadata_filter_values(config)
    filter_match_mode = str(config.get("filter_match", "exact"))

    if not filter_column:
        raise ValueError("taxonomy_holdout requires 'filter_column'.")
    if not species_column:
        raise ValueError("taxonomy_holdout requires 'species_column'.")

    selected_species: set[str] = set()
    selected_species_raw: set[str] = set()
    n_rows = 0
    n_filter_rows = 0

    with open(metadata_csv, "r", encoding="utf-8-sig", newline="", errors="replace") as fh:
        reader = csv.DictReader(fh)
        column_lookup = _metadata_column_lookup(reader.fieldnames)
        if filter_column not in column_lookup:
            raise ValueError(
                f"taxonomy_holdout filter_column {filter_column!r} not found in "
                f"{metadata_csv}. Available columns: {sorted(column_lookup)}"
            )
        if species_column not in column_lookup:
            raise ValueError(
                f"taxonomy_holdout species_column {species_column!r} not found in "
                f"{metadata_csv}. Available columns: {sorted(column_lookup)}"
            )

        filter_key = column_lookup[filter_column]
        species_key = column_lookup[species_column]
        for row in reader:
            n_rows += 1
            if not _matches_metadata_filter(
                row.get(filter_key, ""), filter_values, filter_match_mode
            ):
                continue
            n_filter_rows += 1
            species = row.get(species_key, "").strip()
            if not species:
                continue
            selected_species_raw.add(species)
            selected_species.add(normalize_species_name(species))

    stats = {
        "metadata_csv": str(metadata_csv),
        "filter_column": filter_column,
        "filter_values": filter_values,
        "filter_match": filter_match_mode,
        "species_column": species_column,
        "metadata_rows": n_rows,
        "metadata_filter_rows": n_filter_rows,
        "metadata_species": len(selected_species),
        "metadata_species_examples": sorted(selected_species_raw)[:10],
    }
    return selected_species, stats


def _species_matches_holdout(
    fasta_species: str,
    holdout_species: set[str],
    species_match: str,
) -> bool:
    """Return whether a FASTA species belongs to the holdout species set."""
    if not fasta_species:
        return False
    if species_match == "normalized_exact":
        return normalize_species_name(fasta_species) in holdout_species
    if species_match == "exact":
        return fasta_species.strip() in holdout_species
    if species_match == "casefold":
        return fasta_species.strip().casefold() in holdout_species
    if species_match == "normalized_contains":
        fasta_norm = normalize_species_name(fasta_species)
        return any(
            meta_species in fasta_norm or fasta_norm in meta_species
            for meta_species in holdout_species
        )
    raise ValueError(
        f"Unsupported taxonomy species_match {species_match!r}; expected "
        "'normalized_exact', 'exact', 'casefold', or 'normalized_contains'."
    )


def split_panteon_train_val_records(
    records: list[dict],
    val_fraction: float,
    seed: int,
    superfamily2idx: Optional[dict[str, int]] = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Split mapped PanTEon-label records into train/validation only.

    This is used when the test set is provided externally.  Stratification
    matches the non-taxonomy branch of :func:`split_panteon_records` but does
    not carve an internal test split from ``records``.
    """
    sf_lookup = superfamily2idx or SUPERFAMILY2IDX
    sf_labels = [sf_lookup[r["superfamily"]] for r in records]
    train_recs, val_recs = train_test_split(
        records,
        test_size=val_fraction,
        random_state=seed,
        stratify=sf_labels,
    )
    stats = {
        "taxonomy_holdout_enabled": False,
        "split_mode": "stratified_random_train_val_external_test",
        "total_records": len(records),
        "train_records": len(train_recs),
        "val_records": len(val_recs),
        "test_records": 0,
    }
    return train_recs, val_recs, stats


def split_train_val_records(
    records: list[dict],
    val_fraction: float,
    seed: int,
    superfamily2idx: Optional[dict[str, int]] = None,
    split_mode: str = "stratified_random_train_val_external_test",
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Split already-mapped records into train/validation sets.

    This generic variant is used for combined-source training where records may
    come from both Repbase and PanTEon but share the PanTEon/GLARE 30-label
    vocabulary.  Stratification is by superfamily only; source labels are
    preserved in the returned record dicts for diagnostics.
    """
    sf_lookup = superfamily2idx or SUPERFAMILY2IDX
    sf_labels = [sf_lookup[r["superfamily"]] for r in records]
    train_recs, val_recs = train_test_split(
        records,
        test_size=val_fraction,
        random_state=seed,
        stratify=sf_labels,
    )
    stats = {
        "taxonomy_holdout_enabled": False,
        "split_mode": split_mode,
        "total_records": len(records),
        "train_records": len(train_recs),
        "val_records": len(val_recs),
        "test_records": 0,
    }
    return train_recs, val_recs, stats


def split_stratified_train_val_test_records(
    records: list[dict],
    val_fraction: float,
    seed: int,
    superfamily2idx: Optional[dict[str, int]] = None,
    test_fraction: Optional[float] = None,
    split_mode: str = "stratified_random_train_val_test",
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    """Split mapped records into stratified train/validation/test folds."""
    if test_fraction is None:
        test_fraction = val_fraction
    if not 0 < test_fraction < 1:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}.")
    if not 0 <= val_fraction < 1:
        raise ValueError(
            f"val_fraction must be in [0, 1), got {val_fraction}.")
    if val_fraction + test_fraction >= 1:
        raise ValueError(
            "val_fraction + test_fraction must be < 1 for a non-empty train fold; "
            f"got {val_fraction} + {test_fraction}."
        )

    sf_lookup = superfamily2idx or SUPERFAMILY2IDX
    sf_labels = [sf_lookup[r["superfamily"]] for r in records]
    train_val_recs, test_recs = train_test_split(
        records,
        test_size=test_fraction,
        random_state=seed,
        stratify=sf_labels,
    )

    if val_fraction == 0:
        train_recs = train_val_recs
        val_recs: list[dict] = []
    else:
        train_val_sf = [sf_lookup[r["superfamily"]] for r in train_val_recs]
        relative_val_fraction = val_fraction / (1.0 - test_fraction)
        train_recs, val_recs = train_test_split(
            train_val_recs,
            test_size=relative_val_fraction,
            random_state=seed,
            stratify=train_val_sf,
        )

    stats = {
        "taxonomy_holdout_enabled": False,
        "split_mode": split_mode,
        "total_records": len(records),
        "train_records": len(train_recs),
        "val_records": len(val_recs),
        "test_records": len(test_recs),
        "train_side_records": len(train_recs) + len(val_recs),
        "holdout_records": len(test_recs),
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }
    return train_recs, val_recs, test_recs, stats


def _repbase_stratified_random_split_enabled(data_config: dict[str, Any]) -> bool:
    """Return whether combined Repbase should use an internal random test fold."""
    split_config = data_config.get("repbase_split") or {}
    if isinstance(split_config, str):
        mode = split_config
    else:
        mode = split_config.get("mode", "external_holdout")
    normalized = str(mode).strip().lower().replace("-", "_")
    return normalized in {
        "stratified_random",
        "stratified_random_test",
        "random_stratified",
        "random",
    }


def _with_source(records: list[dict], source: str) -> list[dict]:
    """Return shallow-copied records annotated with a training/eval source."""
    annotated: list[dict] = []
    for rec in records:
        new_rec = dict(rec)
        new_rec["source"] = source
        annotated.append(new_rec)
    return annotated


def _source_counts(records: list[dict]) -> dict[str, int]:
    """Count records by their optional ``source`` annotation."""
    return _counter_dict(Counter(str(r.get("source", "unknown")) for r in records))


def build_combined_repbase_panteon_records(
    data_config: dict[str, Any],
    val_fraction: float,
    seed: int,
    superfamily2idx: Optional[dict[str, int]] = None,
    label_subset_labels: Optional[list[str]] = None,
    log: bool = True,
) -> tuple[list[dict], list[dict], dict[str, list[dict]], dict[str, Any]]:
    """Assemble Repbase+PanTEon train/val records and external tests.

    The combined pool is the union of:
      * Repbase train-side REF records mapped into the configured label space
        from ``data.repbase_dir`` (default ``glare/data/repbase31_filtered``), and
      * PanTEon train-side records. With ``data.taxonomy_holdout`` enabled these
        are the non-taxonomy-holdout records; otherwise they are the train+val
        side of the standard stratified random 80/10/10 PanTEon split, keeping
        the random PanTEon test fold external to combined training.

    External tests remain separate:
      * a PanTEon test set named by ``data.panteon_test_name`` (default
        ``panteon_poales`` for taxonomy holdout, ``panteon_random_test`` for
        no-clade/random split); and
      * a Repbase test set named by ``data.repbase_test_name`` (default
        ``repbase_grass``) from ``data.holdout_dir`` unless
        ``data.repbase_split.mode`` selects ``stratified_random``.  In that
        no-clade/random mode the Repbase test fold is carved from
        ``data.repbase_dir`` and ``data.holdout_dir`` is not read.

    All four pools are filtered by ``label_subset_labels`` only when provided.
    Without a subset, the full configured label vocabulary is retained.
    """
    repbase_train_dir = Path(data_config.get(
        "repbase_dir", "glare/data/repbase31_filtered"))
    repbase_holdout_dir = Path(data_config.get(
        "holdout_dir", "glare/data/repbase31_holdout"))
    toml_path = Path(data_config.get(
        "toml_mapping", "glare/data/repbase-to-repeatmasker.toml"))
    label_space_name = _label_space_name(data_config)
    panteon_label_mapping = data_config.get("panteon_label_mapping")
    panteon_fasta_path = Path(
        data_config.get(
            "panteon_fasta_path",
            data_config.get(
                "fasta_path", "glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta"),
        )
    )
    taxonomy_holdout = data_config.get("taxonomy_holdout")
    taxonomy_holdout_active = taxonomy_holdout_enabled(taxonomy_holdout)
    repbase_random_split_active = _repbase_stratified_random_split_enabled(
        data_config)
    repbase_split_config = data_config.get("repbase_split") or {}
    if isinstance(repbase_split_config, str):
        repbase_split_config = {"mode": repbase_split_config}
    panteon_test_name = str(
        data_config.get(
            "panteon_test_name",
            "panteon_poales" if taxonomy_holdout_active else "panteon_random_test",
        )
    )
    repbase_test_name = str(
        data_config.get(
            "repbase_test_name",
            "repbase_random_test" if repbase_random_split_active else "repbase_grass",
        )
    )

    if _is_custom_repeatmasker_label_space(label_space_name):
        repbase_train_records, repbase_train_diagnostics = load_repbase_dir_for_label_space(
            repbase_train_dir,
            toml_path,
            label_space_name=label_space_name,
        )
        panteon_records = load_panteon_dataset(
            panteon_fasta_path,
            label_mapping_path=Path(
                str(panteon_label_mapping)) if panteon_label_mapping else None,
            label_space_name=label_space_name,
        )
    else:
        repbase_train_records, repbase_train_diagnostics = load_repbase_dir_for_panteon_training(
            repbase_train_dir,
            toml_path,
        )
        panteon_records = load_panteon_dataset(panteon_fasta_path)
    repbase_random_split_stats: dict[str, Any] | None = None
    if repbase_random_split_active:
        repbase_test_fraction = float(
            repbase_split_config.get(
                "test_fraction",
                data_config.get("test_fraction", val_fraction),
            )
        )
        repbase_random_train_recs, repbase_random_val_recs, repbase_test_records, repbase_random_split_stats = (
            split_stratified_train_val_test_records(
                records=repbase_train_records,
                val_fraction=val_fraction,
                seed=seed,
                superfamily2idx=superfamily2idx,
                test_fraction=repbase_test_fraction,
                split_mode="stratified_random_repbase_external_test_plus_combined_train_val",
            )
        )
        repbase_train_records = repbase_random_train_recs + repbase_random_val_recs
        repbase_test_diagnostics = {
            **repbase_train_diagnostics,
            "loaded_records": len(repbase_test_records),
            "mapped_records": len(repbase_test_records),
            "skipped_records": 0,
            "source_loaded_records": repbase_train_diagnostics["loaded_records"],
            "source_mapped_records": repbase_train_diagnostics["mapped_records"],
            "source_skipped_records": repbase_train_diagnostics["skipped_records"],
            "repbase_split_mode": "stratified_random",
            "repbase_split_stats": repbase_random_split_stats,
        }
    elif _is_custom_repeatmasker_label_space(label_space_name):
        repbase_test_records, repbase_test_diagnostics = load_repbase_dir_for_label_space(
            repbase_holdout_dir,
            toml_path,
            label_space_name=label_space_name,
        )
    else:
        repbase_test_records, repbase_test_diagnostics = load_repbase_dir_for_panteon_training(
            repbase_holdout_dir,
            toml_path,
        )
    if taxonomy_holdout_active:
        panteon_train_side_records, panteon_test_records, panteon_partition_stats = (
            partition_panteon_records_by_taxonomy_holdout(
                panteon_records,
                taxonomy_holdout=taxonomy_holdout,
            )
        )
    else:
        panteon_train_recs, panteon_val_recs, panteon_test_records, panteon_random_split_stats = (
            split_panteon_records(
                records=panteon_records,
                val_fraction=val_fraction,
                seed=seed,
                taxonomy_holdout=None,
                superfamily2idx=superfamily2idx,
            )
        )
        panteon_train_side_records = panteon_train_recs + panteon_val_recs
        panteon_partition_stats = {
            **panteon_random_split_stats,
            "split_mode": "stratified_random_panteon_external_test_plus_combined_train_val",
            "train_side_records": len(panteon_train_side_records),
            "holdout_records": len(panteon_test_records),
            "panteon_random_train_records": len(panteon_train_recs),
            "panteon_random_val_records": len(panteon_val_recs),
        }

    subset_stats: dict[str, Any] = {}
    shared_labels_used: list[str] | None = None
    if label_subset_labels is not None:
        shared_labels_used = _ordered_superfamily_labels(label_subset_labels)
        repbase_train_records, subset_stats["repbase_train"] = filter_records_by_superfamilies(
            repbase_train_records,
            shared_labels_used,
            dataset_name="Combined Repbase train source",
            log=log,
        )
        panteon_train_side_records, subset_stats["panteon_train_side"] = filter_records_by_superfamilies(
            panteon_train_side_records,
            shared_labels_used,
            dataset_name="Combined PanTEon train-side source",
            log=log,
        )
        panteon_test_records, subset_stats[f"{panteon_test_name}_test"] = filter_records_by_superfamilies(
            panteon_test_records,
            shared_labels_used,
            dataset_name=f"Combined PanTEon external test [{panteon_test_name}]",
            log=log,
        )
        repbase_test_records, subset_stats[f"{repbase_test_name}_test"] = filter_records_by_superfamilies(
            repbase_test_records,
            shared_labels_used,
            dataset_name=f"Combined Repbase holdout external test [{repbase_test_name}]",
            log=log,
        )

    if not repbase_train_records:
        raise ValueError(
            f"No Repbase train records remain from {repbase_train_dir}.")
    if not panteon_train_side_records:
        raise ValueError(
            f"No PanTEon train-side records remain from {panteon_fasta_path}.")
    if not panteon_test_records:
        raise ValueError(
            "No PanTEon external test records remain after filtering/splitting.")
    if not repbase_test_records:
        repbase_test_source = repbase_train_dir if repbase_random_split_active else repbase_holdout_dir
        raise ValueError(
            f"No Repbase external test records remain from {repbase_test_source}.")

    repbase_train_source = "repbase_random_train_val" if repbase_random_split_active else "repbase_train"
    repbase_train_records = _with_source(
        repbase_train_records, repbase_train_source)
    panteon_train_source = "panteon_non_poales_train" if taxonomy_holdout_active else "panteon_random_train_val"
    panteon_train_side_records = _with_source(
        panteon_train_side_records, panteon_train_source)
    panteon_test_records = _with_source(
        panteon_test_records, f"{panteon_test_name}_test")
    repbase_test_records = _with_source(
        repbase_test_records, f"{repbase_test_name}_test")

    combined_records = repbase_train_records + panteon_train_side_records
    train_recs, val_recs, split_stats = split_train_val_records(
        records=combined_records,
        val_fraction=val_fraction,
        seed=seed,
        superfamily2idx=superfamily2idx,
        split_mode="combined_repbase_panteon_stratified_train_val_external_tests",
    )
    split_stats.update({
        "train_val_records": len(combined_records),
        "external_test_records": len(panteon_test_records) + len(repbase_test_records),
        "panteon_test_name": panteon_test_name,
        "repbase_test_name": repbase_test_name,
        "panteon_test_records": len(panteon_test_records),
        "repbase_test_records": len(repbase_test_records),
        "panteon_poales_test_records": len(panteon_test_records),
        "repbase_grass_test_records": len(repbase_test_records),
        "repbase_train_source_dir": str(repbase_train_dir),
        "repbase_holdout_source_dir": str(repbase_holdout_dir),
        "repbase_test_source_dir": str(repbase_train_dir if repbase_random_split_active else repbase_holdout_dir),
        "repbase_split_mode": "stratified_random" if repbase_random_split_active else "external_holdout",
        "panteon_fasta_path": str(panteon_fasta_path),
    })

    combined_counts = count_record_labels(combined_records)
    train_counts = count_record_labels(train_recs)
    val_counts = count_record_labels(val_recs)
    tests = {
        panteon_test_name: panteon_test_records,
        repbase_test_name: repbase_test_records,
    }
    diagnostics: dict[str, Any] = {
        "mode": "combined_repbase_panteon",
        "repbase_train_dir": str(repbase_train_dir),
        "repbase_holdout_dir": str(repbase_holdout_dir),
        "repbase_test_source_dir": str(repbase_train_dir if repbase_random_split_active else repbase_holdout_dir),
        "repbase_split_mode": "stratified_random" if repbase_random_split_active else "external_holdout",
        "repbase_random_split_active": repbase_random_split_active,
        "repbase_random_split_stats": repbase_random_split_stats,
        "toml_path": str(toml_path),
        "panteon_fasta_path": str(panteon_fasta_path),
        "panteon_label_mapping": str(panteon_label_mapping) if panteon_label_mapping else None,
        "label_space": label_space_name,
        "taxonomy_holdout": taxonomy_holdout,
        "taxonomy_holdout_active": taxonomy_holdout_active,
        "panteon_test_name": panteon_test_name,
        "repbase_test_name": repbase_test_name,
        "shared_labels_used": shared_labels_used,
        "shared_label_count": len(shared_labels_used or []),
        "subset_filter_stats": subset_stats,
        "repbase_train_diagnostics": repbase_train_diagnostics,
        "repbase_test_diagnostics": repbase_test_diagnostics,
        "panteon_total_records": len(panteon_records),
        "panteon_partition_stats": panteon_partition_stats,
        "repbase_train_records_kept": len(repbase_train_records),
        "panteon_train_side_records_kept": len(panteon_train_side_records),
        "combined_records": len(combined_records),
        "train_records": len(train_recs),
        "val_records": len(val_recs),
        "panteon_test_records_kept": len(panteon_test_records),
        "repbase_test_records_kept": len(repbase_test_records),
        "panteon_poales_test_records_kept": len(panteon_test_records),
        "repbase_grass_test_records_kept": len(repbase_test_records),
        "combined_source_counts": _source_counts(combined_records),
        "train_source_counts": _source_counts(train_recs),
        "val_source_counts": _source_counts(val_recs),
        "combined_label_counts": combined_counts,
        "train_label_counts": train_counts,
        "val_label_counts": val_counts,
        "panteon_test_label_counts": count_record_labels(panteon_test_records),
        "repbase_test_label_counts": count_record_labels(repbase_test_records),
        "panteon_poales_test_label_counts": count_record_labels(panteon_test_records),
        "repbase_grass_test_label_counts": count_record_labels(repbase_test_records),
        "split_stats": split_stats,
    }
    return train_recs, val_recs, tests, diagnostics


def _split_panteon_records_with_rare_classes_in_train(
    records: list[dict],
    val_fraction: float,
    seed: int,
    sf_lookup: dict[str, int],
    *,
    error_message: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    """Split PanTEon records when some classes are too rare for 3-way stratification.

    Rare superfamilies are retained in training instead of being dropped.  The
    remaining sufficiently populated classes keep the historical stratified
    80/10/10 behavior.  This fallback is only used when the legacy stratified
    split would fail, preserving existing behavior for well-populated datasets.
    """
    holdout_fraction = val_fraction * 2
    if not 0 < holdout_fraction < 1:
        raise ValueError(
            "2 * val_fraction must be in (0, 1) for a train/validation/test split; "
            f"got val_fraction={val_fraction}."
        )

    counts = Counter(r["superfamily"] for r in records)
    min_count_for_val_and_test = max(2, math.ceil(2 / holdout_fraction))
    rare_superfamilies = {
        sf for sf, count in counts.items() if count < min_count_for_val_and_test
    }
    rare_train_recs = [
        r for r in records if r["superfamily"] in rare_superfamilies]
    stratified_recs = [
        r for r in records if r["superfamily"] not in rare_superfamilies]

    if not stratified_recs:
        raise ValueError(
            "Cannot create validation/test folds: every superfamily is too rare for "
            f"stratification with val_fraction={val_fraction}. Original error: {error_message}"
        )

    stratified_sf = [sf_lookup[r["superfamily"]] for r in stratified_recs]
    train_recs, holdout_recs = train_test_split(
        stratified_recs,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=stratified_sf,
    )
    holdout_sf = [sf_lookup[r["superfamily"]] for r in holdout_recs]
    val_recs, test_recs = train_test_split(
        holdout_recs,
        test_size=0.5,
        random_state=seed,
        stratify=holdout_sf,
    )
    train_recs = train_recs + rare_train_recs

    stats = {
        "taxonomy_holdout_enabled": False,
        "split_mode": "stratified_random_80_10_10_rare_classes_train_only",
        "total_records": len(records),
        "train_records": len(train_recs),
        "val_records": len(val_recs),
        "test_records": len(test_recs),
        "rare_train_only_records": len(rare_train_recs),
        "rare_train_only_superfamilies": sorted(rare_superfamilies),
        "rare_train_only_superfamily_counts": {
            sf: counts[sf] for sf in sorted(rare_superfamilies)
        },
        "min_count_for_val_and_test": min_count_for_val_and_test,
        "fallback_reason": str(error_message),
    }
    return train_recs, val_recs, test_recs, stats


def split_panteon_records(
    records: list[dict],
    val_fraction: float,
    seed: int,
    taxonomy_holdout: Optional[dict[str, Any]] = None,
    superfamily2idx: Optional[dict[str, int]] = None,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    """Split PanTEon records into train/validation/test sets.

    With no active taxonomy holdout, this preserves the historical stratified
    80/10/10 split.  With ``taxonomy_holdout`` enabled, all matching species are
    assigned to test only, and the remaining records are stratified into train
    and validation using ``val_fraction``.
    """
    sf_lookup = superfamily2idx or SUPERFAMILY2IDX
    if not taxonomy_holdout_enabled(taxonomy_holdout):
        sf_labels = [sf_lookup[r["superfamily"]] for r in records]
        try:
            train_recs, holdout_recs = train_test_split(
                records,
                test_size=val_fraction * 2,
                random_state=seed,
                stratify=sf_labels,
            )
            holdout_sf = [sf_lookup[r["superfamily"]] for r in holdout_recs]
            val_recs, test_recs = train_test_split(
                holdout_recs,
                test_size=0.5,
                random_state=seed,
                stratify=holdout_sf,
            )
        except ValueError as exc:
            return _split_panteon_records_with_rare_classes_in_train(
                records,
                val_fraction,
                seed,
                sf_lookup,
                error_message=str(exc),
            )
        stats = {
            "taxonomy_holdout_enabled": False,
            "split_mode": "stratified_random_80_10_10",
            "total_records": len(records),
            "train_records": len(train_recs),
            "val_records": len(val_recs),
            "test_records": len(test_recs),
        }
        return train_recs, val_recs, test_recs, stats

    assert taxonomy_holdout is not None
    holdout_species, metadata_stats = load_taxonomy_holdout_species(
        taxonomy_holdout)
    fail_on_no_matches = bool(taxonomy_holdout.get("fail_on_no_matches", True))
    species_match = str(taxonomy_holdout.get(
        "species_match", "normalized_exact"))
    fasta_species_mode = str(taxonomy_holdout.get(
        "fasta_species_mode", "after_at"))

    if not holdout_species:
        msg = (
            "taxonomy_holdout selected zero species from metadata "
            f"{metadata_stats['metadata_csv']} using "
            f"{metadata_stats['filter_column']}={metadata_stats['filter_values']}."
        )
        if fail_on_no_matches:
            raise ValueError(msg)
        logger.warning(msg)

    train_val_recs, test_recs, partition_stats = partition_panteon_records_by_taxonomy_holdout(
        records,
        taxonomy_holdout=taxonomy_holdout,
    )

    train_val_sf = [sf_lookup[r["superfamily"]] for r in train_val_recs]
    train_recs, val_recs = train_test_split(
        train_val_recs,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_sf,
    )

    train_sf = {r["superfamily"] for r in train_recs}
    test_sf = {r["superfamily"] for r in test_recs}
    absent_test_sf = sorted(test_sf - train_sf)

    stats = {
        **metadata_stats,
        **partition_stats,
        "taxonomy_holdout_enabled": True,
        "split_mode": "taxonomy_holdout_test_plus_stratified_train_val",
        "species_match": species_match,
        "fasta_species_mode": fasta_species_mode,
        "total_records": len(records),
        "train_records": len(train_recs),
        "val_records": len(val_recs),
        "test_records": len(test_recs),
        "non_holdout_records": len(train_val_recs),
        "test_superfamilies_absent_from_train": absent_test_sf,
    }
    return train_recs, val_recs, test_recs, stats


def format_split_stats(stats: dict[str, Any]) -> str:
    """Format split statistics for readable logs."""
    lines = [
        f"split_mode={stats.get('split_mode')}",
        (
            f"records total={stats.get('total_records')}, "
            f"train={stats.get('train_records')}, "
            f"val={stats.get('val_records')}, "
            f"test={stats.get('test_records')}"
        ),
    ]
    if stats.get("train_val_source_dir") or stats.get("test_source_dir"):
        lines.append(
            "sources "
            f"train_val={stats.get('train_val_source_dir')}, "
            f"test={stats.get('test_source_dir')}"
        )
    if stats.get("external_test_records") is not None:
        lines.append(
            f"external_test_records={stats.get('external_test_records')}, "
            f"train_val_records={stats.get('train_val_records')}"
        )
    if stats.get("taxonomy_holdout_enabled"):
        lines.extend([
            (
                "taxonomy filter "
                f"{stats.get('filter_column')}={stats.get('filter_values')} "
                f"from {stats.get('metadata_csv')} "
                f"matched_rows={stats.get('metadata_filter_rows')}/"
                f"{stats.get('metadata_rows')}"
            ),
            (
                f"metadata_species={stats.get('metadata_species')}, "
                f"matched_fasta_species={stats.get('matched_fasta_species')}, "
                f"unmatched_metadata_species={stats.get('unmatched_metadata_species')}, "
                f"missing_fasta_species_records={stats.get('missing_fasta_species_records')}"
            ),
        ])
        absent = stats.get("test_superfamilies_absent_from_train") or []
        if absent:
            lines.append(
                "WARNING test superfamilies absent from train: " +
                ", ".join(absent)
            )
    return "\n".join(lines)


# ===========================================================================
# Legacy Repbase .ref parsing
# ===========================================================================

def parse_ref_file(ref_path: Path) -> list[dict]:
    """Parse a Repbase .ref file into list of {name, category, species, sequence} dicts.

    Header format: ``>ElementName\\tCategory\\tSpecies`` (tab-separated fields).
    Sequence lines follow immediately until the next header or EOF.
    """
    records: list[dict] = []
    current: Optional[dict] = None
    seq_parts: list[str] = []
    record_index = 0

    with open(ref_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Save previous record
                if current is not None:
                    current["sequence"] = "".join(seq_parts).upper()
                    records.append(current)

                # Parse new header
                header = line[1:]  # strip '>'
                parts = header.split("\t")
                name = parts[0].strip() if len(parts) > 0 else ""
                category = parts[1].strip() if len(parts) > 1 else ""
                species = parts[2].strip() if len(parts) > 2 else ""
                current = {
                    "name": name,
                    "category": category,
                    "species": species,
                    "source_file": str(ref_path),
                    "source_index": record_index,
                    "record_id": f"{ref_path}:{record_index}",
                }
                record_index += 1
                seq_parts = []
            else:
                stripped = line.strip()
                if stripped and current is not None:
                    seq_parts.append(stripped)

    # Save last record
    if current is not None:
        current["sequence"] = "".join(seq_parts).upper()
        records.append(current)

    return records


def apply_toml_filter(
    records: list[dict], toml_path: Path,
    order_labels: Optional[list[str]] = None,
    superfamily_labels: Optional[list[str]] = None,
) -> tuple[list[dict], int]:

    # === INÍCIO DO BYPASS DEFINITIVO ===
    filtered = []

    for rec in records:
        new_rec = dict(rec)

        sf = ""
        # Percorre todo o dicionário da sequência (id, description, etc)
        for val in rec.values():
            if isinstance(val, str) and "#Custom/" in val:
                # Extrai a classe que vem logo depois de #Custom/
                sf = val.split("#Custom/")[1].split()[0]
                break

        new_rec["order"] = "Custom"
        new_rec["superfamily"] = sf
        filtered.append(new_rec)

    # Retorna todas as sequências intactas e 0 (zero) descartes
    return filtered, 0
    # === FIM DO BYPASS DEFINITIVO ===


def load_all_ref_files(
    data_dir: Path,
    toml_path: Path,
    order_labels: Optional[list[str]] = None,
    superfamily_labels: Optional[list[str]] = None,
) -> list[dict]:
    """Load and filter all ``.ref`` files from *data_dir* (legacy Repbase workflow).

    Returns a flat list of filtered records (each has ``order``, ``superfamily``).
    """
    all_records: list[dict] = []
    # ref_files = sorted(data_dir.glob("*.ref"))

    # MUDANÇA: Lê qualquer extensão de FASTA e imprime no terminal
    ref_files = [f for f in data_dir.iterdir() if f.name.endswith(
        ('.fa', '.fasta', '.fna', '.ref'))]
    print(
        f"\n[DEBUG GLARE] Encontrados {len(ref_files)} ficheiros na pasta {data_dir}\n")

    for ref_path in ref_files:
        recs = parse_ref_file(ref_path)
        filtered, dropped = apply_toml_filter(
            recs,
            toml_path,
            order_labels=order_labels,
            superfamily_labels=superfamily_labels,
        )
        all_records.extend(filtered)
    return all_records


def map_repeatmasker_to_panteon_sf(rm_label: str) -> str | None:
    """Map a RepeatMasker-style class to a PanTEon/GLARE superfamily token.

    The primary mapping is :data:`REPEATMASKER_TO_PANTEON_SF`, which documents
    all cross-taxonomy reductions considered unambiguous for external Repbase
    evaluation against PanTEon-trained 30-superfamily models.  As a convenience
    for future Repbase files, exact PanTEon tokens and normalized final path
    components are also accepted when they already resolve to one of the 30
    labels.  Ambiguous broad classes deliberately return ``None``.
    """
    label = rm_label.strip()
    if label in AMBIGUOUS_REPEATMASKER_LABELS:
        return None

    explicit = REPEATMASKER_TO_PANTEON_SF.get(label)
    if explicit is not None:
        return explicit

    normalized_full = normalize_sf(label)
    if normalized_full in SUPERFAMILY_LABELS:
        return normalized_full

    if "/" in label:
        normalized_leaf = normalize_sf(label.rsplit("/", 1)[1])
        if normalized_leaf in SUPERFAMILY_LABELS:
            return normalized_leaf

    return None


def map_repbase_records_to_panteon(
    raw_records: list[dict],
    repbase_to_rm: dict[str, str],
) -> tuple[list[dict], dict[str, Any]]:
    """Map parsed Repbase REF records into the PanTEon/GLARE 30-label space.

    Each raw Repbase category is translated through the supplied Repbase →
    RepeatMasker mapping, then reduced with :func:`map_repeatmasker_to_panteon_sf`.
    Ambiguous/unmapped records are skipped and counted rather than coerced.
    """
    mapped_records: list[dict] = []
    raw_category_counts: Counter[str] = Counter()
    repeatmasker_counts: Counter[str] = Counter()
    panteon_sf_counts: Counter[str] = Counter()
    panteon_order_counts: Counter[str] = Counter()
    skipped_reason_counts: Counter[str] = Counter()
    skipped_category_counts: Counter[str] = Counter()
    skipped_repeatmasker_counts: Counter[str] = Counter()
    mapped_class_rows: list[dict] = []

    for rec in raw_records:
        raw_category = rec.get("category", "").strip()
        raw_category_counts[raw_category] += 1

        rm_label = repbase_to_rm.get(raw_category)
        if rm_label is None:
            skipped_reason_counts["missing_repbase_to_repeatmasker_mapping"] += 1
            skipped_category_counts[raw_category] += 1
            continue

        repeatmasker_counts[rm_label] += 1
        sf = map_repeatmasker_to_panteon_sf(rm_label)
        if sf is None:
            skipped_reason_counts["unmapped_repeatmasker_to_panteon_label"] += 1
            skipped_repeatmasker_counts[rm_label] += 1
            continue

        if sf not in SF_TO_ORDER:
            skipped_reason_counts["mapped_label_not_in_panteon_vocab"] += 1
            skipped_repeatmasker_counts[rm_label] += 1
            continue

        order = SF_TO_ORDER[sf]
        panteon_sf_counts[sf] += 1
        panteon_order_counts[order] += 1

        mapped = dict(rec)
        mapped["original_category"] = raw_category
        mapped["repeatmasker_label"] = rm_label
        mapped["superfamily"] = sf
        mapped["order"] = order
        mapped_records.append(mapped)
        mapped_class_rows.append({
            "name": mapped.get("name", ""),
            "species": mapped.get("species", ""),
            "original_category": raw_category,
            "repeatmasker_label": rm_label,
            "panteon_superfamily": sf,
            "panteon_order": order,
            "sequence_length": len(mapped.get("sequence", "")),
        })

    diagnostics: dict[str, Any] = {
        "loaded_records": len(raw_records),
        "mapped_records": len(mapped_records),
        "skipped_records": len(raw_records) - len(mapped_records),
        "raw_category_counts": _counter_dict(raw_category_counts),
        "repeatmasker_counts": _counter_dict(repeatmasker_counts),
        "panteon_superfamily_counts": _counter_dict(panteon_sf_counts),
        "panteon_order_counts": _counter_dict(panteon_order_counts),
        "skipped_reason_counts": _counter_dict(skipped_reason_counts),
        "skipped_category_counts": _counter_dict(skipped_category_counts),
        "skipped_repeatmasker_counts": _counter_dict(skipped_repeatmasker_counts),
        "mapped_class_rows": mapped_class_rows,
        "mapping_rules": dict(sorted(REPEATMASKER_TO_PANTEON_SF.items())),
        "ambiguous_repeatmasker_labels": sorted(AMBIGUOUS_REPEATMASKER_LABELS),
    }
    return mapped_records, diagnostics


def map_records_to_configured_label_space(
    raw_records: list[dict],
    source_to_target: dict[str, str],
    label_space_name: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Map parsed REF records directly into a configured GLARE label space."""
    label_space = get_label_space(label_space_name)
    sf_to_order_map = label_space["sf_to_order"]
    valid_labels = set(label_space["superfamily_labels"])

    mapped_records: list[dict] = []
    raw_category_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    skipped_reason_counts: Counter[str] = Counter()
    skipped_category_counts: Counter[str] = Counter()
    skipped_target_counts: Counter[str] = Counter()
    mapped_class_rows: list[dict] = []

    for rec in raw_records:
        raw_category = rec.get("category", "").strip()
        raw_category_counts[raw_category] += 1

        target_label = source_to_target.get(raw_category)
        if target_label is None:
            skipped_reason_counts["missing_source_to_target_mapping"] += 1
            skipped_category_counts[raw_category] += 1
            continue
        if target_label not in valid_labels:
            skipped_reason_counts["mapped_label_not_in_configured_label_space"] += 1
            skipped_target_counts[target_label] += 1
            continue

        order = str(sf_to_order_map.get(
            target_label, _superfamily_order(target_label)))
        target_counts[target_label] += 1
        order_counts[order] += 1

        mapped = dict(rec)
        mapped["original_category"] = raw_category
        mapped["repeatmasker_label"] = target_label
        mapped["superfamily"] = target_label
        mapped["order"] = order
        mapped_records.append(mapped)
        mapped_class_rows.append({
            "name": mapped.get("name", ""),
            "species": mapped.get("species", ""),
            "original_category": raw_category,
            "repeatmasker_label": target_label,
            "superfamily": target_label,
            "order": order,
            "sequence_length": len(mapped.get("sequence", "")),
        })

    diagnostics: dict[str, Any] = {
        "label_space": label_space["name"],
        "loaded_records": len(raw_records),
        "mapped_records": len(mapped_records),
        "skipped_records": len(raw_records) - len(mapped_records),
        "raw_category_counts": _counter_dict(raw_category_counts),
        "repeatmasker_counts": _counter_dict(target_counts),
        "panteon_superfamily_counts": _counter_dict(target_counts),
        "panteon_order_counts": _counter_dict(order_counts),
        "skipped_reason_counts": _counter_dict(skipped_reason_counts),
        "skipped_category_counts": _counter_dict(skipped_category_counts),
        "skipped_repeatmasker_counts": _counter_dict(skipped_target_counts),
        "mapped_class_rows": mapped_class_rows,
        "mapping_rules": dict(sorted(source_to_target.items())),
        "ambiguous_repeatmasker_labels": [],
    }
    return mapped_records, diagnostics


def load_repbase_dir_for_label_space(
    data_dir: Path,
    toml_path: Path,
    label_space_name: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Load all Repbase REF files under *data_dir* into a configured label space."""
    with open(toml_path, "rb") as fh:
        source_to_target: dict[str, str] = tomllib.load(fh)

    all_raw_records: list[dict] = []
    per_file_rows: list[dict] = []
    for ref_path in sorted(data_dir.glob("*.ref")):
        raw_records = parse_ref_file(ref_path)
        _mapped_records, diagnostics = map_records_to_configured_label_space(
            raw_records,
            source_to_target,
            label_space_name=label_space_name,
        )
        all_raw_records.extend(raw_records)
        per_file_rows.append({
            "file": str(ref_path),
            "loaded_records": diagnostics["loaded_records"],
            "mapped_records": diagnostics["mapped_records"],
            "skipped_records": diagnostics["skipped_records"],
            "mapped_superfamilies": len(diagnostics["panteon_superfamily_counts"]),
        })

    mapped_records, diagnostics = map_records_to_configured_label_space(
        all_raw_records,
        source_to_target,
        label_space_name=label_space_name,
    )
    diagnostics["data_dir"] = str(data_dir)
    diagnostics["toml_path"] = str(toml_path)
    diagnostics["ref_files"] = len(per_file_rows)
    diagnostics["per_file_rows"] = per_file_rows
    return mapped_records, diagnostics


def load_external_repbase_fasta_for_panteon(
    fasta_path: Path,
    toml_path: Path,
) -> tuple[list[dict], dict[str, Any]]:
    """Load external Repbase FASTA/REF input for PanTEon-model evaluation.

    ``fasta_path`` may be either a single Repbase FASTA/REF file or a directory.
    Directory input is loaded as all sorted ``*.ref`` files using the same
    parsing/mapping path as Repbase→PanTEon training data.

    Headers are parsed with :func:`parse_ref_file`, which supports the Repbase
    ``>name<TAB>category<TAB>species`` format observed in ``grasrep.ref`` and
    ``oryrep.ref``.  Each raw Repbase category is translated through
    ``repbase-to-repeatmasker.toml`` and then conservatively reduced into the
    30-label PanTEon/GLARE vocabulary with
    :func:`map_repeatmasker_to_panteon_sf`.

    Records that cannot be mapped are skipped and summarized in the returned
    diagnostics rather than assigned potentially incorrect labels.
    """
    if fasta_path.is_dir():
        return load_repbase_dir_for_panteon_training(
            data_dir=fasta_path,
            toml_path=toml_path,
        )

    raw_records = parse_ref_file(fasta_path)

    with open(toml_path, "rb") as fh:
        repbase_to_rm: dict[str, str] = tomllib.load(fh)

    mapped_records, diagnostics = map_repbase_records_to_panteon(
        raw_records, repbase_to_rm)
    diagnostics["fasta_path"] = str(fasta_path)
    diagnostics["toml_path"] = str(toml_path)
    diagnostics["ref_files"] = 1
    diagnostics["per_file_rows"] = [{
        "file": str(fasta_path),
        "loaded_records": diagnostics["loaded_records"],
        "mapped_records": diagnostics["mapped_records"],
        "skipped_records": diagnostics["skipped_records"],
        "mapped_superfamilies": len(diagnostics["panteon_superfamily_counts"]),
    }]
    return mapped_records, diagnostics


def load_repbase_dir_for_panteon_training(
    data_dir: Path,
    toml_path: Path,
) -> tuple[list[dict], dict[str, Any]]:
    """Load all Repbase REF files under *data_dir* into PanTEon 30 labels."""
    with open(toml_path, "rb") as fh:
        repbase_to_rm: dict[str, str] = tomllib.load(fh)

    all_raw_records: list[dict] = []
    per_file_rows: list[dict] = []
    for ref_path in sorted(data_dir.glob("*.ref")):
        raw_records = parse_ref_file(ref_path)
        mapped_records, diagnostics = map_repbase_records_to_panteon(
            raw_records,
            repbase_to_rm,
        )
        all_raw_records.extend(raw_records)
        per_file_rows.append({
            "file": str(ref_path),
            "loaded_records": diagnostics["loaded_records"],
            "mapped_records": diagnostics["mapped_records"],
            "skipped_records": diagnostics["skipped_records"],
            "mapped_superfamilies": len(diagnostics["panteon_superfamily_counts"]),
        })

    mapped_records, diagnostics = map_repbase_records_to_panteon(
        all_raw_records,
        repbase_to_rm,
    )
    diagnostics["data_dir"] = str(data_dir)
    diagnostics["toml_path"] = str(toml_path)
    diagnostics["ref_files"] = len(per_file_rows)
    diagnostics["per_file_rows"] = per_file_rows
    return mapped_records, diagnostics


# ===========================================================================
# Dataset
# ===========================================================================

class TEDataset(Dataset):
    """PyTorch Dataset for TE sequences.

    Works with both PanTEon records (``superfamily`` is a short token like
    ``"GYPSY"``) and legacy Repbase records (``superfamily`` is a compound
    label like ``"LTR/Gypsy"``).  The label indices are looked up from
    :data:`SUPERFAMILY2IDX` and :data:`ORDER2IDX` which are set by whichever
    ``constants.py`` vocabulary is active.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 1024,
        pad_token_id: int = 0,
        augment: bool = False,
        order2idx: Optional[dict[str, int]] = None,
        superfamily2idx: Optional[dict[str, int]] = None,
        rc_augment: bool = False,
        rc_eval: bool = False,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.order2idx = order2idx or ORDER2IDX
        self.superfamily2idx = superfamily2idx or SUPERFAMILY2IDX
        # NTv3Tokenizer does not return attention_mask; store pad_token_id to
        # construct it manually from input_ids in __getitem__.
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else pad_token_id
        )
        self.augment = augment
        self.rc_augment = rc_augment
        self.rc_eval = rc_eval

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Tokenize a nucleotide context and return model inputs plus labels."""
        rec = self.records[idx]
        sequence = rec["sequence"]

        # Random crop for training augmentation: expose different nucleotide
        # contexts from long sequences. DNABERT-2 and Caduceus apply their
        # HuggingFace tokenizers only after this nucleotide-level crop via their
        # tokenizer wrappers.
        if self.augment and len(sequence) > self.max_length:
            start = random.randint(0, len(sequence) - self.max_length)
            sequence = sequence[start: start + self.max_length]

        # Reverse-complement augmentation (stochastic, training only).
        if self.rc_augment and random.random() < 0.5:
            sequence = sequence.translate(RC_MAP)[::-1]

        # Reverse-complement evaluation (deterministic, always applied when enabled).
        if self.rc_eval:
            sequence = sequence.translate(RC_MAP)[::-1]

        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=128,
        )

        input_ids = encoding["input_ids"].squeeze(0)  # (L,)
        # NTv3 tokenizer may not return attention_mask — construct from pad_token_id
        if "attention_mask" in encoding:
            attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)
        else:
            attention_mask = (input_ids != self.pad_token_id).long()  # (L,)

        order_label = torch.tensor(
            self.order2idx[rec["order"]], dtype=torch.long)
        sf_label = torch.tensor(
            self.superfamily2idx[rec["superfamily"]], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "order_label": order_label,
            "superfamily_label": sf_label,
        }


# ===========================================================================
# Sliding-window Dataset and DataLoader factory
# ===========================================================================

def sequence_windows(
    sequence: str,
    max_length: int,
    overlap: float = 0.5,
) -> list[str]:
    """Return the full-sequence inference windows used by GLARE evaluation.

    A short or exact-length sequence produces one window.  Longer sequences
    use ``max(1, int(max_length * (1 - overlap)))`` as their stride and retain
    the final partial suffix window, matching the historical
    :class:`SlidingWindowDataset` boundaries exactly.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    if len(sequence) <= max_length:
        return [sequence]

    stride = max(1, int(max_length * (1.0 - overlap)))
    windows: list[str] = []
    start = 0
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        windows.append(sequence[start:end])
        if end == len(sequence):
            break
        start += stride
    return windows


class SlidingWindowInferenceDataset(torch.utils.data.Dataset):
    """Unlabelled full-record sliding-window dataset for public inference.

    ``seq_idx`` is the dense source-record position, deliberately independent
    of record IDs so duplicate FASTA identifiers remain separate.  Passing
    ``overlap=None`` produces one full-record context per source record and
    leaves truncation to the tokenizer, preserving the explicit legacy opt-out.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 4096,
        pad_token_id: int = 0,
        overlap: Optional[float] = 0.5,
        rc_eval: bool = False,
    ):
        super().__init__()
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else pad_token_id
        )
        self.n_sequences = len(records)
        self.windows: list[dict] = []

        for seq_idx, rec in enumerate(records):
            sequence = rec["sequence"]
            # RC applies to the complete record before boundaries are generated.
            if rc_eval:
                sequence = sequence.translate(RC_MAP)[::-1]
            record_windows = (
                [sequence]
                if overlap is None
                else sequence_windows(sequence, max_length, overlap)
            )
            self.windows.extend(
                {"seq_idx": seq_idx, "sequence": window}
                for window in record_windows
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        window = self.windows[idx]
        encoding = self.tokenizer(
            window["sequence"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=128,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = (
            encoding["attention_mask"].squeeze(0)
            if "attention_mask" in encoding
            else (input_ids != self.pad_token_id).long()
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_idx": torch.tensor(window["seq_idx"], dtype=torch.long),
        }


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Dataset that expands each sequence into overlapping windows for full-sequence inference.

    For sequences shorter than or equal to ``max_length``, a single window is produced
    (identical to ``TEDataset`` behaviour).  For longer sequences, overlapping windows
    of size ``max_length`` are produced with a stride of ``max_length // 2`` (50% overlap).

    Each item includes a ``seq_idx`` field (int) identifying which original sequence it
    belongs to, so that predictions from multiple windows can be aggregated (averaged)
    back to a single per-sequence prediction.

    Args:
        records: List of record dicts with keys ``sequence``, ``order``, ``superfamily``.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum nucleotide context per window. Character-level
            backbones receive the same number of tokens; DNABERT-2 and Caduceus
            crop to this many nucleotides before HuggingFace tokenization.
        pad_token_id: Token ID used for padding.
        overlap: Fraction of window to overlap between consecutive windows (default 0.5).
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 4096,
        pad_token_id: int = 0,
        overlap: float = 0.5,
        order2idx: Optional[dict[str, int]] = None,
        superfamily2idx: Optional[dict[str, int]] = None,
        rc_eval: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.order2idx = order2idx or ORDER2IDX
        self.superfamily2idx = superfamily2idx or SUPERFAMILY2IDX
        self.rc_eval = rc_eval

        # Build flat list of (seq_idx, window_sequence, order_label, sf_label)
        self.windows: list[dict] = []
        for seq_idx, rec in enumerate(records):
            seq = rec["sequence"]
            order_label = self.order2idx[rec["order"]]
            sf_label = self.superfamily2idx[rec["superfamily"]]

            for window_sequence in sequence_windows(seq, max_length, overlap):
                self.windows.append({
                    "seq_idx": seq_idx,
                    "sequence": window_sequence,
                    "order_label": order_label,
                    "sf_label": sf_label,
                })

        self.n_sequences = len(records)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        sequence = w["sequence"]

        # Preserve historical evaluation semantics: generate forward-strand
        # boundaries first, then reverse-complement each individual window.
        if self.rc_eval:
            sequence = sequence.translate(RC_MAP)[::-1]

        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=128,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        if "attention_mask" in encoding:
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "order_label": torch.tensor(w["order_label"], dtype=torch.long),
            "superfamily_label": torch.tensor(w["sf_label"], dtype=torch.long),
            "seq_idx": torch.tensor(w["seq_idx"], dtype=torch.long),
        }


# ===========================================================================
# PanTEon DataLoader factories
# ===========================================================================

def make_dataloaders(
    fasta_path: Path,
    tokenizer,
    batch_size: int = 16,
    val_fraction: float = 0.1,
    max_length: int = 1024,
    seed: int = 7,
    num_workers: int = 4,
    taxonomy_holdout: Optional[dict[str, Any]] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create stratified train/val :class:`DataLoader` s from a PanTEon FASTA.

    Implements the PanTEon 80/10/10 split:
      1. Split off 20% (val + test) from the full dataset.
      2. The train DataLoader uses the 80% training portion.
      3. The val DataLoader uses 50% of the 20% holdout (i.e. 10% overall).
      4. The remaining 10% (test) is handled separately by
         :func:`make_test_dataloader`.

    Stratification is on ``superfamily`` to ensure every superfamily appears
    in all splits (if possible).

    Args:
        fasta_path: Path to the PanTEon FASTA file.
        tokenizer: NTv3 tokenizer instance.
        batch_size: Mini-batch size.
        val_fraction: Fraction of the full dataset to use as validation
            (default 0.1 → 10%).  The same fraction is used for the test set.
        max_length: Maximum token length for the tokenizer.
        seed: Random seed for reproducibility (PanTEon default: 7).
        num_workers: DataLoader worker processes.

    Returns:
        ``(train_loader, val_loader)`` tuple.
    """
    records = load_panteon_dataset(fasta_path)

    train_recs, val_recs, _test_recs, split_stats = split_panteon_records(
        records=records,
        val_fraction=val_fraction,
        seed=seed,
        taxonomy_holdout=taxonomy_holdout,
    )
    logger.info("PanTEon split:\n%s", format_split_stats(split_stats))

    train_ds = TEDataset(train_recs, tokenizer,
                         max_length=max_length, augment=True)
    val_ds = TEDataset(val_recs, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


def make_test_dataloader(
    fasta_path: Path,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 1024,
    seed: int = 7,
    val_fraction: float = 0.1,
    num_workers: int = 4,
    taxonomy_holdout: Optional[dict[str, Any]] = None,
) -> tuple[DataLoader, list[dict]]:
    """Create a test :class:`DataLoader` from the PanTEon FASTA.

    Uses the same 80/10/10 split logic as :func:`make_dataloaders` to
    reproduce the identical test set.

    Args:
        fasta_path: Path to the PanTEon FASTA file.
        tokenizer: NTv3 tokenizer instance.
        batch_size: Mini-batch size.
        max_length: Maximum token length for the tokenizer.
        seed: Random seed (must match the seed used in :func:`make_dataloaders`).
        val_fraction: Must match the value used in :func:`make_dataloaders`.
        num_workers: DataLoader worker processes.

    Returns:
        ``(test_loader, test_records)`` tuple.  ``test_records`` contains the
        raw record dicts so ground-truth labels can be extracted outside the
        DataLoader.
    """
    records = load_panteon_dataset(fasta_path)

    _train_recs, _val_recs, test_recs, split_stats = split_panteon_records(
        records=records,
        val_fraction=val_fraction,
        seed=seed,
        taxonomy_holdout=taxonomy_holdout,
    )
    logger.info("PanTEon test split:\n%s", format_split_stats(split_stats))

    test_ds = TEDataset(test_recs, tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return test_loader, test_recs


def make_sliding_window_dataloader(
    records: list[dict],
    tokenizer,
    max_length: int,
    pad_token_id: int,
    batch_size: int,
    num_workers: int = 4,
    overlap: float = 0.5,
    order2idx: Optional[dict[str, int]] = None,
    superfamily2idx: Optional[dict[str, int]] = None,
    rc_eval: bool = False,
) -> tuple[torch.utils.data.DataLoader, int]:
    """Build a DataLoader using :class:`SlidingWindowDataset`.

    Returns:
        ``(dataloader, n_sequences)`` — the DataLoader and the number of original
        sequences (needed to pre-allocate the logit accumulation buffer).
    """
    dataset = SlidingWindowDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
        overlap=overlap,
        order2idx=order2idx,
        superfamily2idx=superfamily2idx,
        rc_eval=rc_eval,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, dataset.n_sequences


# ===========================================================================
# Legacy Repbase DataLoader factories
# ===========================================================================

def make_holdout_dataloader(
    holdout_files: list[Path],
    toml_path: Path,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 1024,
    num_workers: int = 4,
) -> tuple[DataLoader, list[dict]]:
    """Create :class:`DataLoader` for a Repbase holdout test set (legacy).

    Also returns the filtered records so ground-truth labels can be extracted
    outside the DataLoader (e.g. for per-sequence result tables).
    """
    all_records: list[dict] = []
    for ref_path in holdout_files:
        recs = parse_ref_file(ref_path)
        filtered, _ = apply_toml_filter(
            recs,
            toml_path,
            order_labels=LEGACY_REPBASE_ORDER_LABELS,
            superfamily_labels=LEGACY_REPBASE_SUPERFAMILY_LABELS,
        )
        all_records.extend(filtered)

    dataset = TEDataset(
        all_records,
        tokenizer,
        max_length=max_length,
        order2idx=LEGACY_REPBASE_ORDER2IDX,
        superfamily2idx=LEGACY_REPBASE_SUPERFAMILY2IDX,
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
    return loader, all_records
