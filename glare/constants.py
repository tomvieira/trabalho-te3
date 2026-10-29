"""Fixed label sets for GLARE — now based on the PanTEon benchmark system.

This module replaces the old Repbase31/TOML-derived 45-superfamily / 7-order
vocabulary with the PanTEon 30-superfamily / 8-order vocabulary used in the
PanTEon benchmark (v1.6.1).

PanTEon reference:
  glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta
  Header format: >ElementName#ClassI/Order/Superfamily  Species name
  Label extraction: id.split(" ")[0].split("#")[1].split("/")[-1] → SF token

Counts: 8 Orders, 30 Superfamilies.

Backward-compatibility note:
  The old Repbase workflow (default.yaml + .ref files) is still supported via
  the legacy constants below, but the primary vocabulary is now PanTEon-based.
  Use panteon.yaml to activate the PanTEon workflow.
"""

import re
from typing import Dict, List

# ── PanTEon superfamily → order mapping ────────────────────────────────────────
# 30 superfamily tokens mapped to their 8 orders
SF_TO_ORDER: Dict[str, str] = {
    "LTR": "LTR",
    "COPIA": "LTR",
    "GYPSY": "LTR",
    "ERV": "LTR",
    "BELPAO": "LTR",
    "LARD": "LTR",
    "LINE": "LINE",
    "I": "LINE",
    "L1": "LINE",
    "RTE": "LINE",
    "CR1": "LINE",
    "R1": "LINE",
    "SINE": "SINE",
    "TRNA": "SINE",
    "ALU": "SINE",
    "PLE": "PLE",
    "DIRS": "DIRS",
    "HELITRON": "HELITRON",
    "CRYPTON": "CRYPTON",
    "HAT": "TIR",
    "MERLIN": "TIR",
    "P": "TIR",
    "TIR": "TIR",
    "TC1MARINER": "TIR",
    "MULE": "TIR",
    "PIFHARBINGER": "TIR",
    "CACTA": "TIR",
    "PIGGYBAC": "TIR",
    "KOLOBOK": "TIR",
    "ACADEM-1": "TIR",
}

# ── Orders ─────────────────────────────────────────────────────────────────────
# 8 TE orders — sorted alphabetically for reproducibility
ORDER_LABELS: List[str] = sorted(set(SF_TO_ORDER.values()))
# ['CRYPTON', 'DIRS', 'HELITRON', 'LINE', 'LTR', 'PLE', 'SINE', 'TIR']

# ── Superfamilies ───────────────────────────────────────────────────────────────
# 30 superfamily tokens — sorted alphabetically for reproducibility
SUPERFAMILY_LABELS: List[str] = sorted(SF_TO_ORDER.keys())
# ['ACADEM-1', 'ALU', 'BELPAO', 'CACTA', 'COPIA', 'CR1', 'CRYPTON', 'DIRS',
#  'ERV', 'GYPSY', 'HAT', 'HELITRON', 'I', 'KOLOBOK', 'L1', 'LARD', 'LINE',
#  'LTR', 'MERLIN', 'MULE', 'P', 'PIFHARBINGER', 'PIGGYBAC', 'PLE', 'R1',
#  'RTE', 'SINE', 'TC1MARINER', 'TIR', 'TRNA']

# ── Index mappings ──────────────────────────────────────────────────────────────
ORDER2IDX: Dict[str, int] = {label: idx for idx, label in enumerate(ORDER_LABELS)}
IDX2ORDER: Dict[int, str] = {idx: label for idx, label in enumerate(ORDER_LABELS)}

SUPERFAMILY2IDX: Dict[str, int] = {
    label: idx for idx, label in enumerate(SUPERFAMILY_LABELS)
}
IDX2SUPERFAMILY: Dict[int, str] = {
    idx: label for idx, label in enumerate(SUPERFAMILY_LABELS)
}

# ── Counts (for config consistency checks) ─────────────────────────────────────
NUM_ORDERS: int = len(ORDER_LABELS)          # 8
NUM_SUPERFAMILIES: int = len(SUPERFAMILY_LABELS)  # 30


# ── Legacy Repbase label vocabulary ───────────────────────────────────────────
#
# The historical Repbase training path used RepeatMasker-style compound labels
# derived from glare/data/repbase-to-repeatmasker.toml.  Keep this fixed list so
# default.yaml can still train/evaluate the 45-superfamily model without
# depending on the active PanTEon 30-superfamily vocabulary above.
LEGACY_REPBASE_SUPERFAMILY_LABELS: List[str] = [
    "DNA/Academ",
    "DNA/CMC",
    "DNA/Crypton",
    "DNA/Dada",
    "DNA/Ginger",
    "DNA/Harbinger",
    "DNA/IS3EU",
    "DNA/Kolobok",
    "DNA/MULE",
    "DNA/Maverick",
    "DNA/Merlin",
    "DNA/Novosib",
    "DNA/P",
    "DNA/PiggyBac",
    "DNA/Sola",
    "DNA/TcMar",
    "DNA/Zator",
    "DNA/Zisupton",
    "DNA/hAT",
    "LINE/CR1",
    "LINE/CRE",
    "LINE/Dong-R4",
    "LINE/Dualen",
    "LINE/I",
    "LINE/L1",
    "LINE/L2",
    "LINE/Proto1",
    "LINE/Proto2",
    "LINE/R1",
    "LINE/R2",
    "LINE/RTE",
    "LINE/Rex-Babar",
    "LINE/Tad1",
    "LTR/Caulimovirus",
    "LTR/Copia",
    "LTR/DIRS",
    "LTR/ERV",
    "LTR/Gypsy",
    "LTR/Pao",
    "PLE/Penelope",
    "RC/Helitron",
    "SINE/5S",
    "SINE/7SL",
    "SINE/U",
    "SINE/tRNA",
]

LEGACY_REPBASE_ORDER_LABELS: List[str] = [
    "DNA",
    "LINE",
    "LTR",
    "Other",
    "PLE",
    "RC",
    "SINE",
]

LEGACY_REPBASE_SF_TO_ORDER: Dict[str, str] = {
    sf: sf.split("/", 1)[0] for sf in LEGACY_REPBASE_SUPERFAMILY_LABELS
}

LEGACY_REPBASE_ORDER2IDX: Dict[str, int] = {
    label: idx for idx, label in enumerate(LEGACY_REPBASE_ORDER_LABELS)
}
LEGACY_REPBASE_IDX2ORDER: Dict[int, str] = {
    idx: label for idx, label in enumerate(LEGACY_REPBASE_ORDER_LABELS)
}

LEGACY_REPBASE_SUPERFAMILY2IDX: Dict[str, int] = {
    label: idx for idx, label in enumerate(LEGACY_REPBASE_SUPERFAMILY_LABELS)
}
LEGACY_REPBASE_IDX2SUPERFAMILY: Dict[int, str] = {
    idx: label for idx, label in enumerate(LEGACY_REPBASE_SUPERFAMILY_LABELS)
}

LEGACY_REPBASE_NUM_ORDERS: int = len(LEGACY_REPBASE_ORDER_LABELS)  # 7
LEGACY_REPBASE_NUM_SUPERFAMILIES: int = len(LEGACY_REPBASE_SUPERFAMILY_LABELS)  # 45


# ── Custom RepeatMasker-style shared label vocabulary ─────────────────────────
#
# Shared target space used by glare/data/panteon-to-repeatmasker-custom.toml and
# glare/data/repbase-to-repeatmasker-custom.toml.  The list is intentionally the
# union of both mapping-file target sets so train/eval checkpoints have a stable
# head size even when experiments use the train-side intersection as a label
# subset.
CUSTOM_REPEATMASKER_SUPERFAMILY_LABELS: List[str] = [
    "CRYPTON/CRYPTON",
    "DIRS/DIRS",
    "HELITRON/HELITRON",
    "LINE/CR1",
    "LINE/I",
    "LINE/L1",
    "LINE/OTHER",
    "LINE/R1",
    "LINE/RTE",
    "LTR/BELPAO",
    "LTR/COPIA",
    "LTR/ERV",
    "LTR/GYPSY",
    "LTR/OTHER",
    "MAVERICK/MAVERICK",
    "PLE/PLE",
    "SATELLITE/SATELLITE",
    "SINE/7SL",
    "SINE/OTHER",
    "SINE/TRNA",
    "STRUCTURAL_RNA/STRUCTURAL_RNA",
    "TIR/ACADEM-1",
    "TIR/CACTA",
    "TIR/HAT",
    "TIR/KOLOBOK",
    "TIR/MERLIN",
    "TIR/MULE",
    "TIR/OTHER",
    "TIR/P",
    "TIR/PIFHARBINGER",
    "TIR/PIGGYBAC",
    "TIR/TC1MARINER",
]

CUSTOM_REPEATMASKER_SF_TO_ORDER: Dict[str, str] = {
    sf: sf.split("/", 1)[0] for sf in CUSTOM_REPEATMASKER_SUPERFAMILY_LABELS
}
CUSTOM_REPEATMASKER_ORDER_LABELS: List[str] = sorted(
    set(CUSTOM_REPEATMASKER_SF_TO_ORDER.values())
)
CUSTOM_REPEATMASKER_ORDER2IDX: Dict[str, int] = {
    label: idx for idx, label in enumerate(CUSTOM_REPEATMASKER_ORDER_LABELS)
}
CUSTOM_REPEATMASKER_IDX2ORDER: Dict[int, str] = {
    idx: label for idx, label in enumerate(CUSTOM_REPEATMASKER_ORDER_LABELS)
}
CUSTOM_REPEATMASKER_SUPERFAMILY2IDX: Dict[str, int] = {
    label: idx for idx, label in enumerate(CUSTOM_REPEATMASKER_SUPERFAMILY_LABELS)
}
CUSTOM_REPEATMASKER_IDX2SUPERFAMILY: Dict[int, str] = {
    idx: label for idx, label in enumerate(CUSTOM_REPEATMASKER_SUPERFAMILY_LABELS)
}
CUSTOM_REPEATMASKER_NUM_ORDERS: int = len(CUSTOM_REPEATMASKER_ORDER_LABELS)  # 11
CUSTOM_REPEATMASKER_NUM_SUPERFAMILIES: int = len(CUSTOM_REPEATMASKER_SUPERFAMILY_LABELS)  # 32


def get_label_space(name: str = "panteon30") -> Dict[str, object]:
    """Return label vocabulary/mappings for a supported GLARE label space."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"panteon", "panteon30", "glare30", "30"}:
        return {
            "name": "panteon30",
            "order_labels": ORDER_LABELS,
            "superfamily_labels": SUPERFAMILY_LABELS,
            "order2idx": ORDER2IDX,
            "idx2order": IDX2ORDER,
            "superfamily2idx": SUPERFAMILY2IDX,
            "idx2superfamily": IDX2SUPERFAMILY,
            "sf_to_order": SF_TO_ORDER,
            "num_orders": NUM_ORDERS,
            "num_superfamilies": NUM_SUPERFAMILIES,
        }
    if normalized in {"legacy", "legacy45", "repbase45", "45"}:
        return {
            "name": "legacy45",
            "order_labels": LEGACY_REPBASE_ORDER_LABELS,
            "superfamily_labels": LEGACY_REPBASE_SUPERFAMILY_LABELS,
            "order2idx": LEGACY_REPBASE_ORDER2IDX,
            "idx2order": LEGACY_REPBASE_IDX2ORDER,
            "superfamily2idx": LEGACY_REPBASE_SUPERFAMILY2IDX,
            "idx2superfamily": LEGACY_REPBASE_IDX2SUPERFAMILY,
            "sf_to_order": LEGACY_REPBASE_SF_TO_ORDER,
            "num_orders": LEGACY_REPBASE_NUM_ORDERS,
            "num_superfamilies": LEGACY_REPBASE_NUM_SUPERFAMILIES,
        }
    if normalized in {"repeatmasker_custom", "custom_repeatmasker", "rm_custom", "repeatmasker32", "32"}:
        return {
            "name": "repeatmasker_custom",
            "order_labels": CUSTOM_REPEATMASKER_ORDER_LABELS,
            "superfamily_labels": CUSTOM_REPEATMASKER_SUPERFAMILY_LABELS,
            "order2idx": CUSTOM_REPEATMASKER_ORDER2IDX,
            "idx2order": CUSTOM_REPEATMASKER_IDX2ORDER,
            "superfamily2idx": CUSTOM_REPEATMASKER_SUPERFAMILY2IDX,
            "idx2superfamily": CUSTOM_REPEATMASKER_IDX2SUPERFAMILY,
            "sf_to_order": CUSTOM_REPEATMASKER_SF_TO_ORDER,
            "num_orders": CUSTOM_REPEATMASKER_NUM_ORDERS,
            "num_superfamilies": CUSTOM_REPEATMASKER_NUM_SUPERFAMILIES,
        }
    raise ValueError(
        f"Unsupported label space {name!r}; expected 'panteon30', 'legacy45', "
        "or 'repeatmasker_custom'."
    )


# ── Derived helpers ─────────────────────────────────────────────────────────────

def sf_to_order(sf_token: str) -> str:
    """Return the order for a PanTEon superfamily token.

    Uses the :data:`SF_TO_ORDER` lookup dict.  This replaces the old
    ``split("/")`` approach used with Repbase compound labels.

    Args:
        sf_token: A PanTEon superfamily token, e.g. ``"GYPSY"`` or ``"L1"``.

    Returns:
        The order string, e.g. ``"LTR"`` or ``"LINE"``.

    Raises:
        KeyError: If *sf_token* is not in :data:`SF_TO_ORDER`.
    """
    return SF_TO_ORDER[sf_token]


def normalize_sf(raw: str) -> str:
    """Normalize a raw superfamily token from a PanTEon FASTA header.

    Applies upper-casing, strips non-alphanumeric characters, and resolves
    known aliases so that the result matches a key in :data:`SF_TO_ORDER`.

    Args:
        raw: Raw superfamily string extracted from the FASTA header, e.g.
            ``"Gypsy"``, ``"LINE-1"``, ``"TC1/Mariner"``.

    Returns:
        Normalized token, e.g. ``"GYPSY"``, ``"L1"``, ``"TC1MARINER"``.
    """
    s = raw.upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s)  # remove non-alphanumeric (hyphens, slashes, etc.)
    # Aliases
    if s == "LINE1":
        s = "L1"
    if s.startswith("TC1"):
        s = "TC1MARINER"
    if s.startswith("PIFHARB"):
        s = "PIFHARBINGER"
    if s.startswith("DIRS"):
        s = "DIRS"
    if s.startswith("ERV"):
        s = "ERV"
    # ACADEM-1 loses its hyphen after stripping → restore
    if s == "ACADEM1":
        s = "ACADEM-1"
    return s
