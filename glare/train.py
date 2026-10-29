"""Training loop for GLARE with DDP support.

Design: dual-head with SF as primary objective.
  - The model outputs both order logits (auxiliary) and superfamily logits (primary).
  - The order head provides a fast, stable hierarchical gradient signal that
    helps the backbone learn a useful TE-order representation early in training.
  - Loss = lambda_sf * sf_loss + lambda_order * order_loss
    where lambda_sf=1.0 >> lambda_order=0.1 (order is auxiliary only).
  - At inference time, ``pred_order`` is derived from ``pred_superfamily`` via
    :func:`~glare.constants.sf_to_order`.
  - Early stopping and checkpointing are based on ``val_sf_f1`` only unless
    ``training.train_on_train_val`` is enabled for final post-HPO fitting.

Supports two data modes:
  - **PanTEon** (new): activated when ``config["data"]["fasta_path"]`` is set.
    Uses :func:`~glare.data.load_panteon_dataset` and a 80/10/10 split.
  - **Repbase** (legacy): activated when ``config["data"]["repbase_dir"]`` is set.
    Uses :func:`~glare.data.load_all_ref_files` with TOML mapping.
"""

import contextlib
import csv
import copy
import ast
import inspect
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from .constants import (
    NUM_ORDERS,
    NUM_SUPERFAMILIES,
    SUPERFAMILY_LABELS,
    ORDER_LABELS,
    SF_TO_ORDER,
    sf_to_order,
    SUPERFAMILY2IDX,
    IDX2SUPERFAMILY,
    ORDER2IDX,
    get_label_space,
)
from .data import (
    TEDataset,
    build_taxonomy_group_test_sets,
    build_combined_repbase_panteon_records,
    filter_panteon_records_by_taxonomy,
    filter_records_by_superfamilies,
    filter_records_by_superfamily_min_count,
    format_label_distribution,
    format_split_stats,
    get_tokenizer,
    make_sliding_window_dataloader,
    resolve_superfamily_label_subset,
    split_panteon_records,
    split_panteon_train_val_records,
)
from .model import build_model, count_parameters

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

INFERENCE_CONFIG_FILENAME = "inference_config.yaml"
LABELS_FILENAME = "labels.json"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "best_model.pt"
UPSTREAM_LICENSE_CAVEAT = (
    "This bundle may include Hugging Face model/tokenizer files saved from the "
    "configured upstream base model. Before uploading, redistributing, or using "
    "it without upstream access, ensure that doing so complies with the upstream "
    "model license and any gating terms."
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: list[int],
    num_classes: int,
    scheme: str = "sqrt_inv",
) -> torch.Tensor:
    """Compute class weights for CrossEntropyLoss (normalised so mean over present classes = 1).

    Args:
        labels: Integer class labels for the training set.
        num_classes: Total number of classes (including absent ones).
        scheme: Weighting scheme — one of:
            ``"none"``     — uniform weight 1.0 for present classes, 0.0 for absent.
            ``"sqrt_inv"`` — square-root inverse frequency (default, softer than pure inv).
            ``"inv"``      — pure inverse frequency (stronger upweighting of rare classes).

    Classes with zero samples receive weight=0 so they are ignored by
    CrossEntropyLoss (as if they did not exist).  The remaining weights
    are normalised so their mean over **present** classes equals 1.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=float)

    if scheme == "none":
        # Uniform weights for present classes
        weights[present] = 1.0
    elif scheme == "inv":
        # Pure inverse frequency: stronger upweighting of rare classes
        weights[present] = 1.0 / counts[present]
        if weights[present].sum() > 0:
            weights[present] = weights[present] / weights[present].mean()
    else:
        # Default: square-root inverse frequency (softer than pure 1/count)
        weights[present] = 1.0 / np.sqrt(counts[present])
        if weights[present].sum() > 0:
            weights[present] = weights[present] / weights[present].mean()

    return torch.tensor(weights, dtype=torch.float32)


def _label_names_for_indices(indices: list[int], label_names: list[str], index_name: str) -> list[str]:
    """Map class indices through the full configured label list with validation."""
    if not indices:
        return []

    n_labels = len(label_names)
    bad_indices = sorted({int(i)
                         for i in indices if int(i) < 0 or int(i) >= n_labels})
    if bad_indices:
        preview = ", ".join(str(i) for i in bad_indices[:10])
        raise ValueError(
            f"{index_name} contain indices outside the configured label space: "
            f"found [{preview}] with valid range 0..{n_labels - 1} "
            f"(num_labels={n_labels})."
        )

    return [label_names[int(i)] for i in indices]


def _validate_logit_dim(logits: torch.Tensor, expected_classes: int, head_name: str) -> None:
    """Raise a clear error if a model head does not match the label space."""
    observed_classes = int(logits.shape[-1])
    if observed_classes != expected_classes:
        raise ValueError(
            f"{head_name} logits have {observed_classes} classes, but the configured "
            f"label space has {expected_classes} classes. Check model.num_* settings "
            "and the active label space."
        )


def _is_main_process() -> bool:
    """Return True if running on DDP rank 0 (or without DDP)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _safe_json_label_mapping(mapping: dict) -> dict[str, object]:
    """Return a JSON-safe copy of a label mapping with stringified keys."""
    return {str(key): value for key, value in mapping.items()}


def _label_space_metadata(label_space: dict) -> dict[str, object]:
    """Serialise the active label space so inference is robust to code changes."""
    return {
        "name": label_space["name"],
        "num_orders": int(label_space["num_orders"]),
        "num_superfamilies": int(label_space["num_superfamilies"]),
        "order_labels": list(label_space["order_labels"]),
        "superfamily_labels": list(label_space["superfamily_labels"]),
        "order2idx": _safe_json_label_mapping(label_space["order2idx"]),
        "idx2order": _safe_json_label_mapping(label_space["idx2order"]),
        "superfamily2idx": _safe_json_label_mapping(label_space["superfamily2idx"]),
        "idx2superfamily": _safe_json_label_mapping(label_space["idx2superfamily"]),
        "sf_to_order": _safe_json_label_mapping(label_space["sf_to_order"]),
    }


def _unwrap_training_model(model: nn.Module, use_ddp: bool) -> nn.Module:
    """Return the raw GLARE model from DDP/torch.compile wrappers."""
    raw_model = model.module if use_ddp else model  # type: ignore[union-attr]
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    return raw_model


def _base_model_family(base_model_name: str) -> str:
    """Infer a stable local subdirectory suffix for the configured backbone."""
    normalized = str(base_model_name).lower().replace("_", "-")
    if "caduceus" in normalized:
        return "caduceus"
    if "plantcad" in normalized:
        return "plantcad"
    if "dnabert-2" in normalized or "dnabert2" in normalized:
        return "dnabert2"
    if "gpn" in normalized:
        return "gpn"
    if "ntv3" in normalized or "instadeepai" in normalized:
        return "ntv3"
    return "backbone"


def _save_pretrained_component(component, save_dir: Path, component_name: str) -> dict[str, object]:
    """Call save_pretrained() if present and return a manifest status entry."""
    save_pretrained = getattr(component, "save_pretrained", None)
    if not callable(save_pretrained):
        return {
            "component": component_name,
            "saved": False,
            "status": "unavailable",
            "message": f"{component_name} does not expose save_pretrained().",
        }

    try:
        save_pretrained(str(save_dir))
        return {
            "component": component_name,
            "saved": True,
            "status": "ok",
            "path": str(save_dir),
        }
    except Exception as exc:  # pragma: no cover - depends on remote model implementations
        return {
            "component": component_name,
            "saved": False,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _candidate_backbone_save_modules(raw_model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return likely Hugging Face backbone objects to persist with save_pretrained()."""
    backbone = getattr(raw_model, "backbone", None)
    candidates: list[tuple[str, nn.Module]] = []

    if backbone is None:
        return candidates

    # For LoRA/PEFT training, save the underlying base model; GLARE-specific
    # heads and adapter weights remain in best_model.pt. Saving the PEFT wrapper
    # itself would only save adapter files and would not remove the upstream base
    # model dependency for inference.
    get_base_model = getattr(backbone, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(("backbone.get_base_model", get_base_model()))
        except Exception:
            pass

    base_model = getattr(backbone, "base_model", None)
    nested_model = getattr(base_model, "model",
                           None) if base_model is not None else None
    if nested_model is not None:
        candidates.append(("backbone.base_model.model", nested_model))

    masked_lm = getattr(backbone, "masked_lm", None)
    if masked_lm is not None:
        candidates.append(("backbone.masked_lm", masked_lm))

    candidates.append(("backbone", backbone))

    unique: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for name, module in candidates:
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        unique.append((name, module))
    return unique


def _save_backbone_pretrained(raw_model: nn.Module, save_dir: Path) -> dict[str, object]:
    """Persist the first available Hugging Face backbone/base model."""
    errors: list[dict[str, object]] = []
    for name, module in _candidate_backbone_save_modules(raw_model):
        status = _save_pretrained_component(module, save_dir, name)
        if status.get("saved"):
            status["selected_component"] = name
            return status
        errors.append(status)
    return {
        "component": "backbone",
        "saved": False,
        "status": "error",
        "attempts": errors,
        "message": "No candidate backbone/base model could be saved with save_pretrained().",
    }


def _dynamic_module_names(save_dir: Path) -> set[str]:
    """Return local Python module names referenced by saved ``auto_map`` data."""
    module_names: set[str] = set()
    for metadata_name in ("config.json", "tokenizer_config.json"):
        metadata_path = save_dir / metadata_name
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for reference in (metadata.get("auto_map") or {}).values():
            references = reference if isinstance(
                reference, list) else [reference]
            for item in references:
                if not isinstance(item, str):
                    continue
                local_reference = item.rsplit("--", 1)[-1]
                module_name, separator, _ = local_reference.rpartition(".")
                if separator and module_name:
                    module_names.add(module_name)
    return module_names


def _component_python_sources(components: list[object]) -> dict[str, Path]:
    """Index importable component/config classes by their source module basename."""
    sources: dict[str, Path] = {}
    for component in components:
        for candidate in (component, getattr(component, "config", None)):
            if candidate is None:
                continue
            try:
                source = Path(inspect.getfile(candidate.__class__)).resolve()
            except (OSError, TypeError):
                continue
            if source.suffix == ".py" and source.is_file():
                sources[source.stem] = source
                # Dynamic-module model/config/tokenizer sources are colocated.
                # Index only Python siblings in that exact authorized module
                # directory; required ``auto_map`` entries decide what is copied.
                for sibling in source.parent.glob("*.py"):
                    if sibling.name != "__init__.py":
                        sources.setdefault(sibling.stem, sibling.resolve())
    return sources


def _relative_python_dependencies(source: Path) -> set[str]:
    """Return sibling module names imported relatively by one Python source file."""
    dependencies: set[str] = set()
    try:
        tree = ast.parse(source.read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return dependencies
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            dependencies.add(node.module.split(".", 1)[0])
    return dependencies


def _localize_saved_auto_maps(save_dir: Path) -> list[str]:
    """Rewrite repository-qualified dynamic references to bundle-local references."""
    updated: list[str] = []
    for metadata_name in ("config.json", "tokenizer_config.json"):
        metadata_path = save_dir / metadata_name
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        auto_map = metadata.get("auto_map") or {}
        localized: dict[str, object] = {}
        changed = False
        for key, reference in auto_map.items():
            if isinstance(reference, list):
                value = [
                    item.rsplit("--", 1)[-1] if isinstance(item, str) else item
                    for item in reference
                ]
            elif isinstance(reference, str):
                value = reference.rsplit("--", 1)[-1]
            else:
                value = reference
            localized[key] = value
            changed = changed or value != reference
        if not changed:
            continue
        metadata["auto_map"] = localized
        temporary = metadata_path.with_suffix(
            metadata_path.suffix + ".partial")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, metadata_path)
        updated.append(metadata_name)
    return updated


def _package_dynamic_module_code(
    *,
    save_dir: Path,
    components: list[object],
) -> dict[str, object]:
    """Package custom Hugging Face Python modules referenced by saved metadata.

    ``save_pretrained()`` does not consistently copy all remote-code files. This
    resolves the exact ``auto_map`` module basenames from already-loaded,
    authorized component classes, follows sibling relative imports, copies only
    those source files, and makes the references local. No Hub lookup is done.
    """
    required = _dynamic_module_names(save_dir)
    if not required:
        return {
            "required": False,
            "status": "not_required",
            "files": [],
            "metadata_updated": [],
        }

    source_index = _component_python_sources(components)
    pending = list(sorted(required))
    resolved: dict[str, Path] = {}
    while pending:
        module_name = pending.pop(0)
        if module_name in resolved:
            continue
        source = source_index.get(module_name)
        if source is None:
            # A loaded remote module's dependencies are normally siblings of its
            # source. Search only those known source directories, never caches at
            # large, to avoid collecting unrelated code.
            for known_source in list(source_index.values()) + list(resolved.values()):
                candidate = known_source.parent / f"{module_name}.py"
                if candidate.is_file():
                    source = candidate.resolve()
                    source_index[module_name] = source
                    break
        if source is None:
            continue
        resolved[module_name] = source
        for dependency in sorted(_relative_python_dependencies(source)):
            if dependency not in resolved:
                pending.append(dependency)

    missing = sorted(required - resolved.keys())
    copied: list[str] = []
    if not missing:
        for module_name, source in sorted(resolved.items()):
            destination = save_dir / f"{module_name}.py"
            temporary = destination.with_suffix(".py.partial")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            copied.append(destination.name)
        metadata_updated = _localize_saved_auto_maps(save_dir)
        return {
            "required": True,
            "status": "ok",
            "files": copied,
            "metadata_updated": metadata_updated,
        }

    return {
        "required": True,
        "status": "error",
        "files": [],
        "metadata_updated": [],
        "missing_modules": missing,
        "message": "Could not resolve every dynamic module from loaded component classes.",
    }


def save_inference_bundle(
    *,
    config: dict,
    output_dir: Path,
    model: nn.Module,
    tokenizer,
    label_space: dict,
    use_ddp: bool,
    checkpoint_filename: str = CHECKPOINT_FILENAME,
) -> None:
    """Write a self-contained GLARE inference bundle next to best_model.pt.

    GLAREModel is a PyTorch module with custom classification heads rather than
    a Hugging Face PreTrainedModel, so this intentionally saves only the
    Hugging Face backbone/base model and tokenizer via save_pretrained().  The
    GLARE heads, pooling, and LoRA adapter state remain in best_model.pt.
    """
    checkpoint_path = output_dir / checkpoint_filename
    manifest_path = output_dir / MANIFEST_FILENAME
    base_model = str(config["model"]["base_model"])
    base_model_dirname = f"base_model_{_base_model_family(base_model)}"
    local_base_model_path = f"./{base_model_dirname}"
    local_base_model_dir = output_dir / base_model_dirname

    manifest: dict[str, object] = {
        "format": "glare_inference_bundle",
        "format_version": 1,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_exists": checkpoint_path.is_file(),
        "inference_config_filename": INFERENCE_CONFIG_FILENAME,
        "labels_filename": LABELS_FILENAME,
        "manifest_filename": MANIFEST_FILENAME,
        "original_base_model": base_model,
        "local_base_model_path": local_base_model_path,
        "tokenizer_save": None,
        "backbone_save": None,
        "dynamic_module_code": None,
        "ood_artifacts": {
            "temperature_scaler": "temperature_scaler.json" if (output_dir / "temperature_scaler.json").is_file() else None,
            "mahal_detector": "mahal_ood.npz" if (output_dir / "mahal_ood.npz").is_file() else None,
        },
        "caveats": [UPSTREAM_LICENSE_CAVEAT],
        "errors": [],
    }

    try:
        from . import __version__ as glare_version
        manifest["glare_version"] = glare_version
    except Exception:
        manifest["glare_version"] = None

    if not checkpoint_path.is_file():
        manifest["errors"].append(
            f"Checkpoint {checkpoint_filename!r} does not exist; skipped Hugging Face artifact saving."
        )
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        logger.warning(
            "Inference bundle not saved because %s is missing.", checkpoint_path)
        return

    raw_model = _unwrap_training_model(model, use_ddp=use_ddp)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        raw_model.load_state_dict(state_dict)
        logger.info(
            "Loaded %s before saving inference bundle artifacts.", checkpoint_path)
    except Exception as exc:
        manifest["errors"].append(
            f"Failed to load checkpoint before bundle save; artifacts may reflect the in-memory model: {exc!r}"
        )

    local_base_model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_status = _save_pretrained_component(
        tokenizer, local_base_model_dir, "tokenizer")
    backbone_status = _save_backbone_pretrained(
        raw_model, local_base_model_dir)
    manifest["tokenizer_save"] = tokenizer_status
    manifest["backbone_save"] = backbone_status
    if not tokenizer_status.get("saved"):
        manifest["errors"].append(
            "Tokenizer could not be saved with save_pretrained(); see tokenizer_save.")
    if not backbone_status.get("saved"):
        manifest["errors"].append(
            "Backbone/base model could not be saved with save_pretrained(); see backbone_save.")

    dynamic_components: list[object] = [tokenizer]
    dynamic_components.extend(
        module for _, module in _candidate_backbone_save_modules(raw_model))
    dynamic_status = _package_dynamic_module_code(
        save_dir=local_base_model_dir,
        components=dynamic_components,
    )
    manifest["dynamic_module_code"] = dynamic_status
    if dynamic_status.get("status") == "error":
        manifest["errors"].append(
            "Required Hugging Face dynamic-module code could not be packaged; "
            "see dynamic_module_code."
        )

    labels_metadata = _label_space_metadata(label_space)
    with open(output_dir / LABELS_FILENAME, "w") as fh:
        json.dump(labels_metadata, fh, indent=2, sort_keys=True)

    inference_config = copy.deepcopy(config)
    inference_config.setdefault("model", {})[
        "base_model"] = local_base_model_path
    inference_config.setdefault(
        "data", {})["label_space"] = str(label_space["name"])
    inference_config["inference_bundle"] = {
        "checkpoint_filename": checkpoint_filename,
        "labels_filename": LABELS_FILENAME,
        "manifest_filename": MANIFEST_FILENAME,
        "local_base_model_path": local_base_model_path,
        "original_base_model": base_model,
    }
    with open(output_dir / INFERENCE_CONFIG_FILENAME, "w") as fh:
        yaml.safe_dump(inference_config, fh, sort_keys=False)

    # Refresh OOD artifact status after all preceding training-time postprocessing.
    manifest["ood_artifacts"] = {
        "temperature_scaler": "temperature_scaler.json" if (output_dir / "temperature_scaler.json").is_file() else None,
        "mahal_detector": "mahal_ood.npz" if (output_dir / "mahal_ood.npz").is_file() else None,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    logger.info(
        "Inference bundle written to %s (config=%s, labels=%s, manifest=%s, local_base_model=%s)",
        output_dir,
        INFERENCE_CONFIG_FILENAME,
        LABELS_FILENAME,
        MANIFEST_FILENAME,
        base_model_dirname,
    )


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    order_criterion: nn.Module,
    sf_criterion: nn.Module,
    device: torch.device,
    lambda_order: float,
    lambda_sf: float,
    gradient_accumulation_steps: int = 1,
    scaler=None,
) -> dict:
    """Train for one epoch with gradient accumulation and BF16 autocast.

    Uses ``model.no_sync()`` for all accumulation steps except the last to
    avoid redundant all-reduce communication across DDP ranks.

    Returns a dict with keys: ``train_loss``, ``order_loss``, ``sf_loss``.
    """
    model.train()
    total_loss = 0.0
    total_order_loss = 0.0
    total_sf_loss = 0.0
    n_batches = 0
    n_loader = len(loader)

    # Zero gradients at the start of the epoch
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        order_labels = batch["order_label"].to(device, non_blocking=True)
        sf_labels = batch["superfamily_label"].to(device, non_blocking=True)

        # Determine if this is the last micro-step in an accumulation window
        is_last_accum = (
            (i + 1) % gradient_accumulation_steps == 0
            or (i + 1) == n_loader
        )

        # Use model.no_sync() for all but the last accumulation step to avoid
        # redundant gradient all-reduce across DDP ranks.
        # Guard with isinstance check: no_sync() only exists on DDP-wrapped models.
        use_no_sync = isinstance(model, DDP) and not is_last_accum
        sync_ctx = model.no_sync() if use_no_sync else contextlib.nullcontext()

        with sync_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                order_logits, sf_logits = model(input_ids, attention_mask)
                o_loss = order_criterion(order_logits, order_labels)
                s_loss = sf_criterion(sf_logits, sf_labels)
                loss = lambda_sf * s_loss + lambda_order * o_loss

            # Scale loss by accumulation steps so gradients are averaged
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

        total_loss += loss.item()
        total_order_loss += o_loss.item()
        total_sf_loss += s_loss.item()
        n_batches += 1

        if is_last_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    denom = max(n_batches, 1)
    return {
        "train_loss": total_loss / denom,
        "order_loss": total_order_loss / denom,
        "sf_loss": total_sf_loss / denom,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    order_criterion: nn.Module,
    sf_criterion: nn.Module,
    device: torch.device,
    lambda_order: float,
    lambda_sf: float,
    epoch: int = 0,
    is_main: bool = True,
    n_sequences: "int | None" = None,
    label_space: dict | None = None,
) -> dict:
    """Evaluate on the validation set.

    Returns a dict with keys: ``val_loss``, ``order_f1``, ``sf_f1``.

    ``order_f1`` is derived from SF predictions via :data:`SF_TO_ORDER` —
    NOT from the auxiliary order head — so it reflects true SF-derived order accuracy.

    If ``n_sequences`` is provided (sliding-window mode), logits from multiple
    windows of the same sequence are averaged before computing predictions.
    The ``loader`` must yield batches with a ``seq_idx`` field in this case.
    """
    from collections import Counter

    model.eval()
    total_loss = 0.0
    n_batches = 0

    label_space = label_space or get_label_space("panteon30")
    sf_label_names = list(label_space["superfamily_labels"])
    order_label_names = list(label_space["order_labels"])
    sf_to_order_map = label_space["sf_to_order"]

    num_sf = len(sf_label_names)
    num_ord = len(order_label_names)

    if n_sequences is not None:
        # Sliding-window mode: accumulate logits per sequence
        sf_logit_sum = torch.zeros(n_sequences, num_sf, device=device)
        ord_logit_sum = torch.zeros(n_sequences, num_ord, device=device)
        window_counts = torch.zeros(n_sequences, device=device)
        sf_labels_store = torch.full(
            (n_sequences,), -1, dtype=torch.long, device=device)
        ord_labels_store = torch.full(
            (n_sequences,), -1, dtype=torch.long, device=device)
    else:
        all_sf_preds: list[int] = []
        all_sf_true: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        order_targets = batch["order_label"].to(device, non_blocking=True)
        sf_targets = batch["superfamily_label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            order_logits, sf_logits = model(input_ids, attention_mask)
            _validate_logit_dim(order_logits, num_ord, "Order")
            _validate_logit_dim(sf_logits, num_sf, "Superfamily")
            o_loss = order_criterion(order_logits, order_targets)
            s_loss = sf_criterion(sf_logits, sf_targets)
            loss = lambda_sf * s_loss + lambda_order * o_loss

        total_loss += loss.item()
        n_batches += 1

        if n_sequences is not None:
            seq_idx = batch["seq_idx"].to(device, non_blocking=True)
            sf_logit_sum.index_add_(0, seq_idx, sf_logits.float())
            ord_logit_sum.index_add_(0, seq_idx, order_logits.float())
            window_counts.index_add_(0, seq_idx, torch.ones(
                seq_idx.size(0), device=device))
            sf_labels_store[seq_idx] = sf_targets
            ord_labels_store[seq_idx] = order_targets
        else:
            all_sf_preds.extend(sf_logits.argmax(dim=-1).cpu().tolist())
            all_sf_true.extend(sf_targets.cpu().tolist())

    denom = max(n_batches, 1)

    if n_sequences is not None:
        counts = window_counts.clamp(min=1).unsqueeze(1)
        sf_avg = sf_logit_sum / counts
        ord_avg = ord_logit_sum / counts
        all_sf_preds = sf_avg.argmax(dim=-1).cpu().tolist()
        all_sf_true = sf_labels_store.cpu().tolist()

    sf_f1 = float(
        f1_score(all_sf_true, all_sf_preds,
                 average="weighted", zero_division=0)
    )

    # Derive order from SF predictions via the full configured label-space lookup.
    # Do not use batch target tensors here: Repbase30 train/val/test splits may
    # contain only a subset of the 30 labels while predictions remain full-space
    # class indices.
    pred_sf_labels = _label_names_for_indices(
        all_sf_preds, sf_label_names, "Predicted superfamily labels"
    )
    true_sf_labels = _label_names_for_indices(
        all_sf_true, sf_label_names, "True superfamily labels"
    )
    pred_order_labels = [sf_to_order_map.get(sf, "") for sf in pred_sf_labels]
    true_order_labels = [sf_to_order_map.get(sf, "") for sf in true_sf_labels]
    order_f1 = float(
        f1_score(true_order_labels, pred_order_labels,
                 average="weighted", zero_division=0)
    )

    # ── DIAGNOSTIC: log SF-derived order distribution every epoch ─────────────
    if is_main:
        pred_counts = Counter(all_sf_preds)
        true_counts = Counter(all_sf_true)
        # Summarise by order
        order_pred_counts: dict[str, int] = {}
        order_true_counts: dict[str, int] = {}
        for idx, sf_label in enumerate(sf_label_names):
            order = sf_to_order_map.get(sf_label, "UNKNOWN")
            order_pred_counts[order] = order_pred_counts.get(
                order, 0) + pred_counts.get(idx, 0)
            order_true_counts[order] = order_true_counts.get(
                order, 0) + true_counts.get(idx, 0)

        per_order_lines = [
            f"    {order}: true={order_true_counts.get(order, 0)}, pred={order_pred_counts.get(order, 0)}"
            for order in sorted(order_true_counts)
        ]
        logger.info(
            "DIAG Epoch %d | order pred distribution (derived from SF, val, n=%d):\n%s",
            epoch, len(all_sf_preds), "\n".join(per_order_lines),
        )

        per_class_sf_f1 = f1_score(
            all_sf_true, all_sf_preds,
            labels=list(range(len(sf_label_names))),
            average=None, zero_division=0,
        )
        # Log the 10 worst SF classes
        worst = sorted(
            [(sf_label_names[i], per_class_sf_f1[i])
             for i in range(len(sf_label_names))],
            key=lambda x: x[1],
        )[:10]
        logger.info(
            "DIAG Epoch %d | 10 lowest SF per-class F1:\n    %s",
            epoch, "\n    ".join(f"{name}={f1:.4f}" for name, f1 in worst),
        )
    # ──────────────────────────────────────────────────────────────────────────

    return {
        "val_loss": total_loss / denom,
        "order_f1": order_f1,
        "sf_f1": sf_f1,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict, output_dir: Path, local_rank: int = 0):
    """Main training function with optional DDP support.

    Detects data mode from config:
      - **PanTEon mode**: ``config["data"]["fasta_path"]`` is set.
      - **Repbase mode** (legacy): ``config["data"]["repbase_dir"]`` is set.

    Args:
        config: Nested dict loaded from a YAML config file.
        output_dir: Directory to write checkpoints and logs.
        local_rank: GPU index for this process (set by torchrun via env var).
    """
    # ── DDP init ────────────────────────────────────────────────────────────
    use_ddp = dist.is_available() and "RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = (global_rank == 0)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("GLARE training started — output dir: %s", output_dir)

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = config["training"]["seed"]
    torch.manual_seed(seed + global_rank)
    np.random.seed(seed + global_rank)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    base_model = config["model"]["base_model"]
    max_length = config["model"]["max_length"]

    if is_main:
        logger.info("Loading tokenizer from %s", base_model)
    tokenizer = get_tokenizer(base_model)

    # ── Data ────────────────────────────────────────────────────────────────
    batch_size = config["training"]["batch_size"]
    val_fraction = config["training"]["val_fraction"]
    train_on_train_val = bool(
        config["training"].get("train_on_train_val", False))
    num_workers = min(4, os.cpu_count() or 4)

    # Gradient accumulation steps (default=1 for backward compat)
    gradient_accumulation_steps = config["training"].get(
        "gradient_accumulation_steps", 1)

    # Detect data mode
    data_config = config.get("data", {})
    data_mode = str(data_config.get("mode", "")).strip().lower()
    combined_mode = data_mode in {
        "combined_repbase_panteon",
        "repbase_panteon_combined",
    } or bool(data_config.get("combined", {}).get("enabled", False))
    panteon_mode = (not combined_mode) and "fasta_path" in data_config
    configured_label_space = str(
        data_config.get("label_space")
        or data_config.get("repbase_label_space", "legacy45")
    )
    configured_label_space_norm = configured_label_space.lower().replace("-", "_")
    repbase_mapped_mode = (not combined_mode) and (not panteon_mode) and configured_label_space_norm in {
        "panteon",
        "panteon30",
        "glare30",
        "30",
        "repeatmasker_custom",
        "custom_repeatmasker",
        "rm_custom",
        "repeatmasker32",
        "32",
    }
    label_space_name_for_training = (
        configured_label_space
        if (combined_mode or panteon_mode or repbase_mapped_mode)
        else "legacy45"
    )
    label_space = get_label_space(label_space_name_for_training)
    order_labels = label_space["order_labels"]
    sf_labels = label_space["superfamily_labels"]
    order2idx = label_space["order2idx"]
    superfamily2idx = label_space["superfamily2idx"]
    sf_to_order_map = label_space["sf_to_order"]

    expected_orders = int(label_space["num_orders"])
    expected_sfs = int(label_space["num_superfamilies"])
    # if config["model"]["num_orders"] != expected_orders or config["model"]["num_superfamilies"] != expected_sfs:
    # raise ValueError(
    #    "Config model class counts do not match label space "
    #    f"{label_space['name']}: config has "
    #    f"num_orders={config['model']['num_orders']}, "
    #    f"num_superfamilies={config['model']['num_superfamilies']}; "
    #    f"expected {expected_orders}/{expected_sfs}."
    # )

    label_subset_labels, label_subset_stats = resolve_superfamily_label_subset(
        config.get("data", {})
    )
    if is_main and label_subset_labels is not None:
        logger.info(
            "Superfamily label subset enabled: mode=%s labels=%d [%s]",
            (label_subset_stats or {}).get("mode"),
            len(label_subset_labels),
            ",".join(label_subset_labels),
        )
        if label_subset_stats:
            logger.info(
                "Superfamily train-intersection excluded labels: Repbase-only=%s; PanTEon-only=%s",
                label_subset_stats.get("repbase_only_labels_excluded", []),
                label_subset_stats.get("panteon_only_labels_excluded", []),
            )

    test_sets: dict[str, list[dict]] = {}

    if combined_mode:
        # ── Combined Repbase + PanTEon shared-label mode ─────────────────────
        if is_main:
            repbase_split_mode = (
                (data_config.get("repbase_split") or {}).get(
                    "mode", "external_holdout")
                if isinstance(data_config.get("repbase_split") or {}, dict)
                else data_config.get("repbase_split", "external_holdout")
            )
            logger.info(
                "Combined Repbase+PanTEon mode: Repbase train=%s; Repbase test source=%s; Repbase split=%s; PanTEon FASTA=%s",
                data_config.get(
                    "repbase_dir", "glare/data/repbase31_filtered"),
                data_config.get("holdout_dir", data_config.get(
                    "repbase_dir", "glare/data/repbase31_filtered")),
                repbase_split_mode,
                data_config.get(
                    "panteon_fasta_path",
                    data_config.get(
                        "fasta_path", "glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta"),
                ),
            )

        train_recs, val_recs, test_sets, combined_diagnostics = build_combined_repbase_panteon_records(
            data_config=data_config,
            val_fraction=val_fraction,
            seed=seed,
            superfamily2idx=superfamily2idx,
            label_subset_labels=label_subset_labels,
            log=is_main,
        )
        test_recs = []  # final evaluation uses named external test_sets below

        if is_main:
            shared_labels = combined_diagnostics.get(
                "shared_labels_used") or []
            logger.info(
                "Combined shared labels used (%d): %s",
                len(shared_labels),
                ",".join(shared_labels),
            )
            if label_subset_stats:
                logger.info(
                    "Combined excluded labels: Repbase-only=%s; PanTEon-only=%s",
                    label_subset_stats.get("repbase_only_labels_excluded", []),
                    label_subset_stats.get("panteon_only_labels_excluded", []),
                )
            rep_diag = combined_diagnostics["repbase_train_diagnostics"]
            rep_test_diag = combined_diagnostics["repbase_test_diagnostics"]
            logger.info(
                "Combined Repbase train source parse/mapping (%s): loaded=%d mapped=%d skipped=%d ref_files=%d shared_label_kept=%d",
                combined_diagnostics["repbase_train_dir"],
                rep_diag["loaded_records"],
                rep_diag["mapped_records"],
                rep_diag["skipped_records"],
                rep_diag["ref_files"],
                combined_diagnostics["repbase_train_records_kept"],
            )
            logger.info(
                "Combined Repbase train skipped reasons: %s",
                rep_diag["skipped_reason_counts"],
            )
            logger.info(
                "Combined PanTEon partition (%s; taxonomy_holdout_active=%s): total=%d train_side=%d external_test=%d train_kept=%d test_kept=%d",
                combined_diagnostics["panteon_partition_stats"].get(
                    "split_mode"),
                combined_diagnostics.get("taxonomy_holdout_active"),
                combined_diagnostics["panteon_total_records"],
                combined_diagnostics["panteon_partition_stats"].get(
                    "train_side_records"),
                combined_diagnostics["panteon_partition_stats"].get(
                    "holdout_records"),
                combined_diagnostics["panteon_train_side_records_kept"],
                combined_diagnostics["panteon_poales_test_records_kept"],
            )
            logger.info(
                "Combined Repbase external test parse/mapping (%s; split=%s): loaded=%d mapped=%d skipped=%d ref_files=%d shared_label_kept=%d",
                combined_diagnostics["repbase_test_source_dir"],
                combined_diagnostics.get("repbase_split_mode"),
                rep_test_diag["loaded_records"],
                rep_test_diag["mapped_records"],
                rep_test_diag["skipped_records"],
                rep_test_diag["ref_files"],
                combined_diagnostics["repbase_grass_test_records_kept"],
            )
            logger.info(
                "Combined pool source counts: %s; train source counts: %s; val source counts: %s",
                combined_diagnostics["combined_source_counts"],
                combined_diagnostics["train_source_counts"],
                combined_diagnostics["val_source_counts"],
            )
            logger.info(
                "Combined split sizes: combined_pool=%d train=%d val=%d %s=%d %s=%d",
                combined_diagnostics["combined_records"],
                combined_diagnostics["train_records"],
                combined_diagnostics["val_records"],
                combined_diagnostics["panteon_test_name"],
                combined_diagnostics["panteon_test_records_kept"],
                combined_diagnostics["repbase_test_name"],
                combined_diagnostics["repbase_test_records_kept"],
            )
            logger.info("Combined split:\n%s", format_split_stats(
                combined_diagnostics["split_stats"]))
            logger.info("%s", format_label_distribution(
                train_recs, "Combined train distribution"))
            logger.info("%s", format_label_distribution(
                val_recs, "Combined validation distribution"))
            for test_name, records_for_test in test_sets.items():
                logger.info("%s", format_label_distribution(
                    records_for_test, f"Combined external test distribution [{test_name}]"))

    elif panteon_mode:
        # ── PanTEon mode ────────────────────────────────────────────────────
        from .data import load_panteon_dataset

        fasta_path = Path(config["data"]["fasta_path"])
        if is_main:
            logger.info("PanTEon mode: loading data from %s", fasta_path)

        panteon_label_mapping = data_config.get("panteon_label_mapping")
        records = load_panteon_dataset(
            fasta_path,
            label_mapping_path=Path(
                str(panteon_label_mapping)) if panteon_label_mapping else None,
            label_space_name=label_space["name"],
        )
        if label_subset_labels is not None:
            records, subset_filter_stats = filter_records_by_superfamilies(
                records,
                label_subset_labels,
                dataset_name="PanTEon all records before split",
                log=is_main,
            )
            if not records:
                raise ValueError(
                    "data.label_subset removed all PanTEon records before splitting."
                )
        taxonomy_train_filter = data_config.get(
            "taxonomy_train_filter") or data_config.get("taxonomy_filter")
        records, taxonomy_train_filter_stats = filter_panteon_records_by_taxonomy(
            records,
            taxonomy_train_filter,
            filter_name="data.taxonomy_train_filter",
        )
        if is_main and taxonomy_train_filter_stats.get("taxonomy_filter_enabled"):
            logger.info(
                "PanTEon taxonomy train filter enabled: %s=%s kept=%d/%d metadata_species=%s matched_fasta_species=%s",
                taxonomy_train_filter_stats.get("filter_column"),
                taxonomy_train_filter_stats.get("filter_values"),
                taxonomy_train_filter_stats.get("matched_records"),
                taxonomy_train_filter_stats.get("total_records"),
                taxonomy_train_filter_stats.get("metadata_species"),
                taxonomy_train_filter_stats.get("matched_fasta_species"),
            )
        if not records:
            raise ValueError(
                "data.taxonomy_train_filter removed all PanTEon records before splitting.")
        min_superfamily_count = data_config.get("min_superfamily_count")
        records, min_count_stats = filter_records_by_superfamily_min_count(
            records,
            min_superfamily_count,
            dataset_name="PanTEon records after taxonomy train filter",
            log=is_main,
        )
        if min_count_stats.get("enabled") and is_main:
            logger.info(
                "PanTEon min superfamily count filter enabled: min_count=%s kept_superfamilies=%d kept_records=%d skipped=%d",
                min_count_stats.get("min_count"),
                min_count_stats.get("kept_superfamily_count"),
                min_count_stats.get("kept_records"),
                min_count_stats.get("skipped_records"),
            )
        if not records:
            raise ValueError(
                "data.min_superfamily_count removed all PanTEon records before splitting.")
        if is_main:
            logger.info("Total filtered PanTEon records: %d", len(records))

        train_recs, val_recs, test_recs, split_stats = split_panteon_records(
            records=records,
            val_fraction=val_fraction,
            seed=seed,
            taxonomy_holdout=config.get("data", {}).get("taxonomy_holdout"),
            superfamily2idx=superfamily2idx,
        )

        if is_main:
            logger.info("PanTEon split:\n%s", format_split_stats(split_stats))
        if test_recs:
            include_default_test_set = bool(
                data_config.get("include_default_test_set", True))
            test_sets = {
                "panteon_test": test_recs} if include_default_test_set else {}
            taxonomy_test_groups = data_config.get(
                "taxonomy_test_groups") or []
            if taxonomy_test_groups:
                grouped_test_sets, taxonomy_group_stats = build_taxonomy_group_test_sets(
                    test_recs,
                    taxonomy_test_groups,
                )
                test_sets.update(grouped_test_sets)
                if is_main:
                    logger.info(
                        "PanTEon taxonomy test groups enabled: %d groups from %d held-out test records",
                        len(grouped_test_sets),
                        taxonomy_group_stats["total_records"],
                    )
                    for test_name, group_records in grouped_test_sets.items():
                        group_stats = taxonomy_group_stats["groups"].get(
                            test_name, {})
                        logger.info(
                            "PanTEon taxonomy test group [%s]: n=%d metadata_filter=%s=%s metadata_species=%s matched_fasta_species=%s",
                            test_name,
                            len(group_records),
                            group_stats.get("filter_column"),
                            group_stats.get("filter_values"),
                            group_stats.get("metadata_species"),
                            group_stats.get("matched_fasta_species"),
                        )

    elif repbase_mapped_mode:
        # ── Repbase REF → configured shared-label mode ───────────────────────
        from .data import load_repbase_dir_for_label_space, load_repbase_dir_for_panteon_training

        train_val_dir = Path(config["data"]["repbase_dir"])
        holdout_dir = Path(config["data"]["holdout_dir"])
        toml_path = Path(config["data"]["toml_mapping"])

        if is_main:
            logger.info(
                "Repbase→%s mode: train/validation source=%s; external test source=%s; mapping=%s",
                label_space["name"],
                train_val_dir,
                holdout_dir,
                toml_path,
            )

        if label_space["name"] == "panteon30":
            train_val_records, train_val_diagnostics = load_repbase_dir_for_panteon_training(
                train_val_dir,
                toml_path,
            )
        else:
            train_val_records, train_val_diagnostics = load_repbase_dir_for_label_space(
                train_val_dir,
                toml_path,
                label_space_name=str(label_space["name"]),
            )
        if label_subset_labels is not None:
            train_val_records, train_val_subset_stats = filter_records_by_superfamilies(
                train_val_records,
                label_subset_labels,
                dataset_name=f"Repbase→{label_space['name']} train/validation source",
                log=is_main,
            )
        if not train_val_records:
            raise ValueError(
                f"No Repbase records from {train_val_dir} mapped into {label_space['name']} label space."
            )

        if label_space["name"] == "panteon30":
            test_recs, holdout_diagnostics = load_repbase_dir_for_panteon_training(
                holdout_dir,
                toml_path,
            )
        else:
            test_recs, holdout_diagnostics = load_repbase_dir_for_label_space(
                holdout_dir,
                toml_path,
                label_space_name=str(label_space["name"]),
            )
        if label_subset_labels is not None:
            test_recs, test_subset_stats = filter_records_by_superfamilies(
                test_recs,
                label_subset_labels,
                dataset_name=f"Repbase→{label_space['name']} external test source",
                log=is_main,
            )
        if not test_recs:
            raise ValueError(
                f"No Repbase holdout records from {holdout_dir} mapped into {label_space['name']} label space."
            )

        train_recs, val_recs, split_stats = split_panteon_train_val_records(
            records=train_val_records,
            val_fraction=val_fraction,
            seed=seed,
            superfamily2idx=superfamily2idx,
        )
        split_stats.update({
            "total_records": len(train_val_records) + len(test_recs),
            "train_val_records": len(train_val_records),
            "test_records": len(test_recs),
            "external_test_records": len(test_recs),
            "train_val_source_dir": str(train_val_dir),
            "test_source_dir": str(holdout_dir),
        })

        if is_main:
            logger.info(
                "Repbase→%s train/validation parse/mapping summary (%s): loaded=%d mapped=%d skipped=%d ref_files=%d",
                label_space["name"],
                train_val_dir,
                train_val_diagnostics["loaded_records"],
                train_val_diagnostics["mapped_records"],
                train_val_diagnostics["skipped_records"],
                train_val_diagnostics["ref_files"],
            )
            logger.info(
                "Repbase→%s train/validation skipped reasons: %s",
                label_space["name"],
                train_val_diagnostics["skipped_reason_counts"],
            )
            logger.info(
                "Repbase→%s train/validation mapped order counts: %s",
                label_space["name"],
                train_val_diagnostics["panteon_order_counts"],
            )
            logger.info(
                "Repbase→%s train/validation mapped superfamily counts: %s",
                label_space["name"],
                train_val_diagnostics["panteon_superfamily_counts"],
            )
            logger.info(
                "Repbase→%s external test parse/mapping summary (%s): loaded=%d mapped=%d skipped=%d ref_files=%d",
                label_space["name"],
                holdout_dir,
                holdout_diagnostics["loaded_records"],
                holdout_diagnostics["mapped_records"],
                holdout_diagnostics["skipped_records"],
                holdout_diagnostics["ref_files"],
            )
            logger.info(
                "Repbase→%s external test skipped reasons: %s",
                label_space["name"],
                holdout_diagnostics["skipped_reason_counts"],
            )
            logger.info(
                "Repbase→%s external test mapped order counts: %s",
                label_space["name"],
                holdout_diagnostics["panteon_order_counts"],
            )
            logger.info(
                "Repbase→%s external test mapped superfamily counts: %s",
                label_space["name"],
                holdout_diagnostics["panteon_superfamily_counts"],
            )
            logger.info(
                "Repbase→%s final split sizes: train=%d val=%d external_test=%d (train/val source=%s; test source=%s)",
                label_space["name"],
                len(train_recs),
                len(val_recs),
                len(test_recs),
                train_val_dir,
                holdout_dir,
            )
            logger.info("Repbase→%s split:\n%s",
                        label_space["name"], format_split_stats(split_stats))
            logger.info("%s", format_label_distribution(
                train_recs, f"Repbase→{label_space['name']} train distribution"))
            logger.info("%s", format_label_distribution(
                val_recs, f"Repbase→{label_space['name']} validation distribution"))
            logger.info("%s", format_label_distribution(
                test_recs, f"Repbase→{label_space['name']} external test distribution"))
        if test_recs:
            test_sets = {"repbase_grass": test_recs}

    else:
        # ── Legacy Repbase mode ─────────────────────────────────────────────
        from sklearn.model_selection import train_test_split
        from .data import load_all_ref_files

        data_dir = Path(config["data"]["repbase_dir"])
        toml_path = Path(config["data"]["toml_mapping"])

        if is_main:
            logger.info(
                "Repbase mode: loading training data from %s", data_dir)

        # === INÍCIO DO BYPASS DE CLASSES E PASTAS ===
        pasta_treino = data_dir / "Train"  # A barra '/' mantém o caminho como objeto Path
        order_labels = ["Custom"]
        # Extrai os nomes limpos (sem o .fa)
        sf_labels = sorted([f.stem for f in pasta_treino.glob("*.fa")])
        # === FIM DO BYPASS DE CLASSES ===

        records = load_all_ref_files(
            pasta_treino,
            toml_path,
            order_labels=list(order_labels),
            superfamily_labels=list(sf_labels),
        )

        if is_main:
            logger.info(
                "Total filtered legacy Repbase records: %d", len(records))
            logger.info("%s", format_label_distribution(
                records, "Legacy Repbase filtered distribution"))

        sf_labels_all = [superfamily2idx[r["superfamily"]] for r in records]
        train_recs, val_recs = train_test_split(
            records,
            test_size=val_fraction,
            random_state=seed,
            stratify=sf_labels_all,
        )
        test_recs = []  # no separate test set in Repbase mode

    # Optional final-training mode: after all mode-specific splitting has
    # already carved out independent test sets, fold the validation split back
    # into training.  This is intended for post-HPO final fitting with selected
    # hyperparameters; it deliberately disables validation-based early stopping
    # and best-checkpoint selection below.
    if train_on_train_val:
        original_train_records = len(train_recs)
        original_val_records = len(val_recs)
        train_recs = list(train_recs) + list(val_recs)
        val_recs = []
        if is_main:
            logger.info(
                "Final train_on_train_val mode enabled: moved %d validation records into training "
                "after preserving held-out test sets; train %d -> %d, val -> 0. "
                "Validation metrics, early stopping, validation checkpoint selection, and "
                "temperature scaling will be disabled.",
                original_val_records,
                original_train_records,
                len(train_recs),
            )
            logger.info("%s", format_label_distribution(
                train_recs, "Final train+validation distribution"))

    # ── Build datasets ───────────────────────────────────────────────────────
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    sliding_window_overlap = config.get(
        "data", {}).get("sliding_window_overlap", 0.5)

    train_ds = TEDataset(
        train_recs,
        tokenizer,
        max_length=max_length,
        augment=True,
        order2idx=order2idx,
        superfamily2idx=superfamily2idx,
        rc_augment=data_config.get("rc_augment", False),
    )

    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=global_rank, shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )

    val_loader = None
    n_val_sequences = None
    if not train_on_train_val:
        if not val_recs:
            raise ValueError(
                "Validation split is empty. Set training.train_on_train_val: true "
                "only for final fixed-epoch fitting, or use a non-empty validation split."
            )
        val_loader, n_val_sequences = make_sliding_window_dataloader(
            val_recs, tokenizer, max_length, pad_token_id,
            batch_size=batch_size,
            num_workers=num_workers,
            overlap=sliding_window_overlap,
            order2idx=order2idx,
            superfamily2idx=superfamily2idx,
            rc_eval=data_config.get("rc_eval", False),
        )

    # ── Class weights ────────────────────────────────────────────────────────
    order_labels_train = [order2idx[r["order"]] for r in train_recs]
    sf_labels_train = [superfamily2idx[r["superfamily"]] for r in train_recs]

    num_orders = config["model"]["num_orders"]
    num_superfamilies = config["model"]["num_superfamilies"]

    class_weight_scheme = config["training"].get(
        "class_weight_scheme", "sqrt_inv")
    order_weights = compute_class_weights(
        order_labels_train, num_orders, scheme=class_weight_scheme).to(device)
    sf_weights = compute_class_weights(
        sf_labels_train, num_superfamilies, scheme=class_weight_scheme).to(device)

    label_smoothing = config["training"]["label_smoothing"]
    order_criterion = nn.CrossEntropyLoss(
        weight=order_weights, label_smoothing=label_smoothing
    )
    sf_criterion = nn.CrossEntropyLoss(
        weight=sf_weights, label_smoothing=label_smoothing
    )

    # ── DIAGNOSTIC: log training-set order and SF label distributions ─────
    if is_main:
        from collections import Counter
        train_order_counts = Counter(order_labels_train)
        diag_lines = [
            f"    {order_labels[i]}: {train_order_counts.get(i, 0)}"
            for i in range(num_orders)
        ]
        logger.info(
            "DIAG Train set order distribution (n=%d):\n%s",
            len(order_labels_train), "\n".join(diag_lines),
        )
        logger.info(
            "DIAG Order class weights: %s",
            {order_labels[i]: f"{order_weights[i].item():.4f}" for i in range(
                num_orders)},
        )

        # SF label distribution
        train_sf_counts = Counter(sf_labels_train)
        sf_diag_lines = [
            f"    SF {i} ({sf_labels[i]}): count={train_sf_counts.get(i, 0)}, weight={sf_weights[i].item():.4f}"
            for i in range(num_superfamilies)
        ]
        logger.info(
            "DIAG Train set SF label distribution (n=%d):\n%s",
            len(sf_labels_train), "\n".join(sf_diag_lines),
        )
    # ──────────────────────────────────────────────────────────────────────

    lambda_order = float(config["training"]["lambda_order"])
    lambda_sf = float(config["training"]["lambda_superfamily"])

    # ── Model ────────────────────────────────────────────────────────────────
    if is_main:
        logger.info("Building model…")
    model = build_model(config)
    model = model.to(device)

    # ── torch.compile (PyTorch 2.x) ──────────────────────────────────────────
    use_compile = config["training"].get("use_compile", False)
    if use_compile:
        try:
            if is_main:
                logger.info("Applying torch.compile(mode='default')…")
            model = torch.compile(model, mode="default")
            if is_main:
                logger.info("torch.compile applied successfully.")
        except Exception as exc:
            if is_main:
                logger.warning("torch.compile failed (skipping): %s", exc)

    ddp_find_unused_parameters = use_ddp and (
        lambda_order <= 0.0 or lambda_sf <= 0.0
    )
    if use_ddp:
        if is_main and ddp_find_unused_parameters:
            logger.info(
                "DDP unused-parameter detection enabled because one task loss weight is zero "
                "(lambda_order=%.4f, lambda_superfamily=%.4f).",
                lambda_order,
                lambda_sf,
            )
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=ddp_find_unused_parameters,
        )

    if is_main:
        # Unwrap compile + DDP to get the raw GLAREModel for parameter counting
        raw_model_for_count = model
        if use_ddp:
            # type: ignore[union-attr]
            raw_model_for_count = raw_model_for_count.module
        # torch.compile wraps in _OptimizedModule; unwrap if needed
        if hasattr(raw_model_for_count, "_orig_mod"):
            raw_model_for_count = raw_model_for_count._orig_mod
        trainable, total = count_parameters(raw_model_for_count)
        backbone_trainable = sum(
            p.numel() for p in raw_model_for_count.backbone.parameters() if p.requires_grad
        )
        backbone_total = sum(p.numel()
                             for p in raw_model_for_count.backbone.parameters())
        head_trainable = sum(
            p.numel() for p in raw_model_for_count.head.parameters() if p.requires_grad
        )
        head_total = sum(p.numel()
                         for p in raw_model_for_count.head.parameters())
        finetune_mode = (
            "LoRA adapter fine-tuning"
            if raw_model_for_count.use_lora
            else "full backbone fine-tuning"
        )
        logger.info("Fine-tuning mode: %s", finetune_mode)
        logger.info(
            "Parameters — trainable: %s / total: %s (%.2f%%); "
            "backbone trainable: %s / %s (%.2f%%); head trainable: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
            f"{backbone_trainable:,}",
            f"{backbone_total:,}",
            100.0 * backbone_trainable / backbone_total,
            f"{head_trainable:,}",
            f"{head_total:,}",
            100.0 * head_trainable / head_total,
        )

    # ── DIAGNOSTIC: log trainable backbone module names ───────────────────
    if is_main:
        # type: ignore[union-attr]
        raw_model_diag = model.module if use_ddp else model
        if hasattr(raw_model_diag, "_orig_mod"):
            raw_model_diag = raw_model_diag._orig_mod
        backbone_trainable_named = [
            n for n, p in raw_model_diag.backbone.named_parameters() if p.requires_grad
        ]
        unique_parents = sorted(
            {".".join(n.split(".")[:-1]) for n in backbone_trainable_named})
        mode_label = "LoRA" if raw_model_diag.use_lora else "full fine-tuning backbone"
        logger.info(
            "DIAG %s trainable modules (%d param tensors, %d unique layers):\n    %s",
            mode_label,
            len(backbone_trainable_named), len(unique_parents),
            "\n    ".join(unique_parents[:20]) +
            (" ..." if len(unique_parents) > 20 else ""),
        )
    # ──────────────────────────────────────────────────────────────────────

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    backbone_lr = float(config["training"]["lora_lr"])
    head_lr = float(config["training"]["head_lr"])
    weight_decay = config["training"]["weight_decay"]

    raw_model = model.module if use_ddp else model  # type: ignore[union-attr]
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # Separate trainable backbone params (LoRA adapters or full backbone) from head params.
    backbone_params = [
        p for n, p in raw_model.backbone.named_parameters() if p.requires_grad
    ]
    head_params = list(raw_model.head.parameters())

    if is_main:
        backbone_group = "LoRA adapter" if raw_model.use_lora else "full backbone"
        logger.info(
            "Optimizer: %s params use training.lora_lr=%g; head params use training.head_lr=%g",
            backbone_group,
            backbone_lr,
            head_lr,
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    max_epochs = config["training"]["max_epochs"]
    warmup_fraction = config["training"]["warmup_fraction"]
    save_checkpoint = config["training"].get("save_checkpoint", True)

    # With gradient accumulation, the scheduler steps once per optimizer step
    # (not per batch). With DDP, len(train_loader) is already divided by world_size
    # (DistributedSampler), so we multiply back by world_size to get the correct
    # total number of optimizer steps across the full training run.
    total_steps = (len(train_loader) //
                   gradient_accumulation_steps) * max_epochs
    warmup_steps = int(total_steps * warmup_fraction)

    if is_main:
        logger.info(
            "Scheduler: total_optimizer_steps=%d, warmup_steps=%d, "
            "gradient_accumulation_steps=%d",
            total_steps, warmup_steps, gradient_accumulation_steps,
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # BF16 does not need GradScaler; autocast is applied inside train_epoch.
    scaler = None

    # ── Training loop ────────────────────────────────────────────────────────
    patience = config["training"]["early_stopping_patience"]

    best_sf_f1 = -1.0
    epochs_no_improve = 0
    log_rows: list[dict] = []
    last_epoch = 0

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        if use_ddp:
            train_sampler.set_epoch(epoch)  # type: ignore[possibly-undefined]

        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            order_criterion, sf_criterion, device,
            lambda_order, lambda_sf,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scaler=scaler,
        )
        elapsed = time.time() - t0

        if train_on_train_val:
            val_metrics = {
                "val_loss": "",
                "order_f1": "",
                "sf_f1": "",
            }
        else:
            assert val_loader is not None
            val_metrics = eval_epoch(
                model, val_loader, order_criterion, sf_criterion,
                device, lambda_order, lambda_sf,
                epoch=epoch, is_main=is_main,
                n_sequences=n_val_sequences,
                label_space=label_space,
            )

        row = {
            "epoch": epoch,
            "phase": "train_val_final" if train_on_train_val else "train_val",
            "test_dataset": "",
            **train_metrics,
            **val_metrics,
            "elapsed_s": round(elapsed, 1),
            "test_accuracy": "",
            "test_n_total": "",
        }
        log_rows.append(row)

        if is_main and train_on_train_val:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | no validation (train_on_train_val final fitting) | %.1fs",
                epoch, max_epochs,
                train_metrics["train_loss"],
                elapsed,
            )
        elif is_main:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "order_f1=%.4f | sf_f1=%.4f | %.1fs",
                epoch, max_epochs,
                train_metrics["train_loss"], val_metrics["val_loss"],
                val_metrics["order_f1"], val_metrics["sf_f1"],
                elapsed,
            )

        if train_on_train_val:
            should_stop = False
        else:
            current_sf_f1 = float(val_metrics["sf_f1"])

            if is_main and current_sf_f1 > best_sf_f1:
                best_sf_f1 = current_sf_f1
                epochs_no_improve = 0
                if save_checkpoint:
                    # Unwrap DDP and compile to get raw state dict
                    # type: ignore[union-attr]
                    raw_model_ckpt = model.module if use_ddp else model
                    if hasattr(raw_model_ckpt, "_orig_mod"):
                        raw_model_ckpt = raw_model_ckpt._orig_mod
                    checkpoint = {
                        "model_state_dict": raw_model_ckpt.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_sf_f1": best_sf_f1,
                        "config": config,
                    }
                    ckpt_path = output_dir / "best_model.pt"
                    torch.save(checkpoint, ckpt_path)
                    logger.info(
                        "  ✓ New best val sf_f1=%.4f — checkpoint saved.", best_sf_f1)
                else:
                    logger.info(
                        "  ✓ New best val sf_f1=%.4f — new best (checkpoint saving disabled).", best_sf_f1)
            elif not is_main:
                # Non-main ranks track improve count for barrier sync
                if current_sf_f1 > best_sf_f1:
                    best_sf_f1 = current_sf_f1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                epochs_no_improve += 1

            if use_ddp:
                # Sync early-stopping decision across ranks
                stop_tensor = torch.tensor(
                    1 if epochs_no_improve >= patience else 0,
                    device=device,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() == 1
            else:
                should_stop = epochs_no_improve >= patience

        if should_stop:
            if is_main:
                logger.info("Early stopping triggered after %d epochs.", epoch)
            break

    if train_on_train_val and is_main:
        if save_checkpoint:
            # type: ignore[union-attr]
            raw_model_ckpt = model.module if use_ddp else model
            if hasattr(raw_model_ckpt, "_orig_mod"):
                raw_model_ckpt = raw_model_ckpt._orig_mod
            checkpoint = {
                "model_state_dict": raw_model_ckpt.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": last_epoch,
                "val_sf_f1": None,
                "selection_metric": "final_epoch_no_validation",
                "train_on_train_val": True,
                "config": config,
            }
            ckpt_path = output_dir / "best_model.pt"
            torch.save(checkpoint, ckpt_path)
            logger.info(
                "Final train+validation model saved to %s after fixed %d/%d epochs; "
                "no validation metric was used for checkpoint selection.",
                ckpt_path,
                last_epoch,
                max_epochs,
            )
        else:
            logger.info(
                "Final train+validation fitting completed after %d/%d epochs; "
                "checkpoint saving disabled by training.save_checkpoint.",
                last_epoch,
                max_epochs,
            )

    # ── Save training log ────────────────────────────────────────────────────
    if is_main and log_rows:
        log_path = output_dir / "train_log.csv"
        fieldnames = list(log_rows[0].keys())
        with open(log_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
        logger.info("Training log saved to %s", log_path)

    # ── PanTEon-label test-set evaluation after training ─────────────────────
    if is_main and (combined_mode or panteon_mode or repbase_mapped_mode) and test_sets:
        checkpoint_description = "final train+validation checkpoint" if train_on_train_val else "best checkpoint"
        logger.info(
            "Running %s test-set evaluation on %s for %d dataset(s)…",
            label_space["name"],
            checkpoint_description,
            len(test_sets),
        )
        try:
            from .evaluate import evaluate_panteon_test

            # Load selected checkpoint: validation-best for standard training, final epoch for train+val fitting.
            ckpt_path = output_dir / "best_model.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                # Unwrap DDP and compile to get raw model for loading
                # type: ignore[union-attr]
                raw_model_eval = model.module if use_ddp else model
                if hasattr(raw_model_eval, "_orig_mod"):
                    raw_model_eval = raw_model_eval._orig_mod
                raw_model_eval.load_state_dict(ckpt["model_state_dict"])
                raw_model_eval.to(device)

            # Use the raw (unwrapped) model for test evaluation
            # type: ignore[union-attr]
            raw_model_eval = model.module if use_ddp else model
            if hasattr(raw_model_eval, "_orig_mod"):
                raw_model_eval = raw_model_eval._orig_mod

            test_rows: list[dict] = []
            for test_name, records_for_test in test_sets.items():
                test_loader, n_test_sequences = make_sliding_window_dataloader(
                    records_for_test, tokenizer, max_length, pad_token_id,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    overlap=sliding_window_overlap,
                    order2idx=order2idx,
                    superfamily2idx=superfamily2idx,
                )

                test_metrics = evaluate_panteon_test(
                    raw_model_eval,
                    test_loader,
                    device,
                    config,
                    n_sequences=n_test_sequences,
                )

                logger.info(
                    "%s TEST [%s] | sf_f1=%.4f | order_f1=%.4f | accuracy=%.4f | n=%d",
                    label_space["name"],
                    test_name,
                    test_metrics["sf_f1"],
                    test_metrics["order_f1"],
                    test_metrics["accuracy"],
                    test_metrics["n_total"],
                )

                test_rows.append({
                    "epoch": "TEST",
                    "phase": "test",
                    "test_dataset": test_name,
                    "train_loss": "",
                    "order_loss": "",
                    "sf_loss": "",
                    "val_loss": "",
                    "order_f1": test_metrics["order_f1"],
                    "sf_f1": test_metrics["sf_f1"],
                    "elapsed_s": "",
                    "test_accuracy": test_metrics["accuracy"],
                    "test_n_total": test_metrics["n_total"],
                })

            log_path = output_dir / "train_log.csv"
            with open(log_path, "a", newline="") as fh:
                fieldnames = list(log_rows[0].keys()) if log_rows else list(
                    test_rows[0].keys())
                writer = csv.DictWriter(
                    fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writerows(test_rows)
            logger.info("Test metrics appended to %s", log_path)

        except Exception as exc:
            logger.warning("Test-set evaluation failed: %s", exc)

    # ── OOD calibration (TemperatureScaler + MahalanobisOOD) ─────────────────
    fit_calibration = config.get("ood", {}).get("fit_calibration", True)
    if is_main and fit_calibration:
        if train_on_train_val:
            logger.info(
                "Fitting validation-independent OOD artifacts for train_on_train_val final model: "
                "TemperatureScaler is skipped because no validation set exists; MahalanobisOOD will use train+validation embeddings."
            )
        else:
            logger.info(
                "Fitting OOD calibration artifacts (TemperatureScaler + MahalanobisOOD)…")
        try:
            from .ood import TemperatureScaler, MahalanobisOOD

            # Reload best checkpoint into a clean raw model for calibration
            ckpt_path = output_dir / "best_model.pt"
            # type: ignore[union-attr]
            raw_model_cal = model.module if use_ddp else model
            if hasattr(raw_model_cal, "_orig_mod"):
                raw_model_cal = raw_model_cal._orig_mod
            if ckpt_path.exists():
                ckpt_cal = torch.load(ckpt_path, map_location="cpu")
                raw_model_cal.load_state_dict(ckpt_cal["model_state_dict"])
            raw_model_cal.eval()
            raw_model_cal.to(device)

            # ── Temperature Scaling: collect val logits + labels ──────────────
            if train_on_train_val:
                stale_scaler_path = output_dir / "temperature_scaler.json"
                if stale_scaler_path.exists():
                    stale_scaler_path.unlink()
                    logger.warning(
                        "Removed stale TemperatureScaler artifact at %s because final train+validation fitting has no validation split.",
                        stale_scaler_path,
                    )
                logger.warning(
                    "Skipping TemperatureScaler fitting: training.train_on_train_val=True leaves no validation split."
                )
            else:
                assert val_loader is not None
                all_val_logits: list[torch.Tensor] = []
                all_val_labels: list[torch.Tensor] = []
                with torch.no_grad():
                    for batch in val_loader:
                        b_input_ids = batch["input_ids"].to(
                            device, non_blocking=True)
                        b_attn_mask = batch["attention_mask"].to(
                            device, non_blocking=True)
                        b_sf_labels = batch["superfamily_label"]
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda"),
                        ):
                            _ord_logits, sf_logits_cal = raw_model_cal(
                                b_input_ids, b_attn_mask)
                        all_val_logits.append(sf_logits_cal.float().cpu())
                        all_val_labels.append(b_sf_labels.cpu())

                val_logits_cat = torch.cat(all_val_logits, dim=0)
                val_labels_cat = torch.cat(all_val_labels, dim=0)

                scaler = TemperatureScaler()
                fitted_T = scaler.fit(val_logits_cat, val_labels_cat)
                scaler_path = output_dir / "temperature_scaler.json"
                scaler.save(str(scaler_path))
                logger.info("Temperature scaler saved to %s (T=%.4f)",
                            scaler_path, fitted_T)

            # ── Mahalanobis OOD: collect train trunk embeddings ───────────────
            if hasattr(raw_model_cal, "extract_trunk_embeddings"):
                all_train_emb: list[np.ndarray] = []
                all_train_labels: list[np.ndarray] = []
                with torch.no_grad():
                    for batch in train_loader:
                        b_input_ids = batch["input_ids"].to(
                            device, non_blocking=True)
                        b_attn_mask = batch["attention_mask"].to(
                            device, non_blocking=True)
                        b_sf_labels = batch["superfamily_label"].numpy()
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda"),
                        ):
                            trunk_emb = raw_model_cal.extract_trunk_embeddings(
                                b_input_ids, b_attn_mask
                            )
                        all_train_emb.append(trunk_emb.float().cpu().numpy())
                        all_train_labels.append(b_sf_labels)

                train_emb_np = np.concatenate(all_train_emb, axis=0)
                train_labels_np = np.concatenate(all_train_labels, axis=0)

                mahal = MahalanobisOOD()
                mahal.fit(train_emb_np, train_labels_np)
                mahal_path = output_dir / "mahal_ood.npz"
                mahal.save(str(mahal_path))
                logger.info(
                    "Mahalanobis OOD detector saved to %s (%d classes)",
                    mahal_path, len(mahal.class_ids_),
                )
            else:
                logger.warning(
                    "Model does not expose extract_trunk_embeddings(); "
                    "skipping MahalanobisOOD fitting."
                )

        except Exception as exc:
            logger.warning(
                "OOD calibration failed (non-fatal): %s", exc, exc_info=True)

    if is_main:
        try:
            save_inference_bundle(
                config=config,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                label_space=label_space,
                use_ddp=use_ddp,
            )
        except Exception as exc:
            logger.warning(
                "Inference bundle saving failed (non-fatal): %s", exc, exc_info=True)

    if use_ddp:
        dist.destroy_process_group()
