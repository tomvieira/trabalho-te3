"""Click CLI for GLARE."""

import copy
import math
import os
import queue
import shutil
import sys
import tempfile
import time
import traceback
from importlib import metadata
from pathlib import Path

import click
import torch
import yaml


DEFAULT_HF_REPO_ID = "Dichopsis/GLARE"
DEFAULT_HF_CHECKPOINT_FILENAME = "best_model.pt"
INFERENCE_CONFIG_FILENAME = "inference_config.yaml"
LABELS_FILENAME = "labels.json"
TEMPERATURE_SCALER_FILENAME = "temperature_scaler.json"
MAHAL_OOD_FILENAME = "mahal_ood.npz"
DEFAULT_PREDICT_CONFIG_FILENAME = (
    "combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_"
    "best_hpo_final.yaml"
)
DEFAULT_PREDICT_CONFIG_DISPLAY = f"glare/configs/{DEFAULT_PREDICT_CONFIG_FILENAME}"
DEFAULT_PREDICTION_CSV_FILENAME = "predictions.csv"
_ROTARY_CACHE_STATE_DICT_SUFFIXES = (
    ".rotary_embedding.cos_cached",
    ".rotary_embedding.sin_cached",
)


def _strip_rotary_embedding_cache_buffers(state_dict):
    """Remove non-parameter rotary embedding cache buffers from a checkpoint."""
    cache_keys = [
        key
        for key in state_dict.keys()
        if key.endswith(_ROTARY_CACHE_STATE_DICT_SUFFIXES)
    ]
    if not cache_keys:
        return state_dict, []

    sanitized_state_dict = copy.copy(state_dict)
    for key in cache_keys:
        sanitized_state_dict.pop(key)
    return sanitized_state_dict, cache_keys


def _resolve_hf_checkpoint(
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str | None,
) -> Path:
    """Download or locate the default prediction checkpoint from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise click.ClickException(
            "No local --checkpoint was provided, so GLARE needs to download a "
            "checkpoint from Hugging Face. Install the optional dependency with "
            "`pip install huggingface_hub` or reinstall GLARE with its project "
            "dependencies, then retry. Alternatively, pass --checkpoint PATH."
        ) from exc

    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        revision_msg = f" at revision {revision!r}" if revision else ""
        raise click.ClickException(
            "Failed to download GLARE checkpoint from Hugging Face "
            f"repo {repo_id!r}, file {filename!r}{revision_msg}: {exc}. "
            "Pass --checkpoint PATH to use a local checkpoint instead, or "
            "override --hf-repo-id/--hf-filename/--hf-revision as needed."
        ) from exc

    return Path(checkpoint_path)


def _download_optional_hf_artifact(
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str | None,
    display_name: str,
) -> Path | None:
    """Best-effort download of an optional prediction artifact from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        artifact_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception:
        return None

    resolved = Path(artifact_path)
    click.echo(f"Using Hugging Face {display_name}: {resolved}")
    return resolved


def _resolve_predict_checkpoint(
    checkpoint: str | None,
    hf_repo_id: str,
    hf_filename: str,
    hf_revision: str | None,
    hf_cache_dir: str | None,
) -> Path:
    """Resolve the prediction checkpoint from a local path or Hugging Face."""
    if checkpoint:
        return Path(checkpoint)

    resolved = _resolve_hf_checkpoint(
        repo_id=hf_repo_id,
        filename=hf_filename,
        revision=hf_revision,
        cache_dir=hf_cache_dir,
    )
    click.echo(
        "Using Hugging Face checkpoint "
        f"{hf_repo_id}/{hf_filename} from cache path: {resolved}"
    )
    return resolved


def _distribution_config_candidates(filename: str) -> list[Path]:
    """Return config paths installed as distribution data files, if available."""
    try:
        distribution = metadata.distribution("glare")
    except metadata.PackageNotFoundError:
        return []

    candidates: list[Path] = []
    for file in distribution.files or []:
        normalized = str(file).replace("\\", "/")
        if normalized.endswith(f"glare/configs/{filename}") or normalized.endswith(
            f"configs/{filename}"
        ):
            candidates.append(Path(distribution.locate_file(file)))
    return candidates


def _default_predict_config_candidates() -> list[Path]:
    """Return cwd-independent candidate paths for the bundled predict config."""
    package_dir = Path(__file__).resolve().parent
    repo_dir = package_dir.parent
    filename = DEFAULT_PREDICT_CONFIG_FILENAME

    candidates = [
        # Source/editable repository layout: <repo>/configs/*.yaml
        repo_dir / "configs" / filename,
        # Future-proof package-data layout: <site-packages>/glare/configs/*.yaml
        package_dir / "configs" / filename,
        # Target installs may place data-files beside the package tree.
        repo_dir / "share" / "glare" / "configs" / filename,
        # Wheel data-files layout configured in pyproject.toml.
        Path(sys.prefix) / "share" / "glare" / "configs" / filename,
        Path(sys.base_prefix) / "share" / "glare" / "configs" / filename,
        *_distribution_config_candidates(filename),
    ]

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def _resolve_predict_config(config: str | None) -> Path:
    """Resolve predict config while preserving explicit --config path behavior."""
    if config is not None:
        return Path(config)

    candidates = _default_predict_config_candidates()
    for candidate in candidates:
        if candidate.is_file():
            click.echo(f"Using default prediction config: {candidate}")
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise click.ClickException(
        "Default prediction config was not found in the installed GLARE package "
        "or source repository. Pass --config PATH to use a custom config. "
        f"Searched:\n  {searched}"
    )


def _resolve_predict_config_for_checkpoint(
    config: str | None,
    checkpoint_path: Path,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    allow_hf_bundle_config: bool = False,
) -> Path:
    """Auto-use sibling inference_config.yaml for local bundles when possible."""
    if config is not None:
        return _resolve_predict_config(config)

    bundle_config = checkpoint_path.parent / INFERENCE_CONFIG_FILENAME
    if bundle_config.is_file():
        click.echo(f"Using bundled inference config: {bundle_config}")
        return bundle_config

    if allow_hf_bundle_config and hf_repo_id:
        hf_config = _download_optional_hf_artifact(
            repo_id=hf_repo_id,
            filename=INFERENCE_CONFIG_FILENAME,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            display_name="bundled inference config",
        )
        if hf_config is not None:
            return hf_config

    return _resolve_predict_config(None)


def _resolve_evaluate_config_for_checkpoint(
    config: str | None,
    checkpoint_path: Path,
) -> Path:
    """Resolve evaluation config, preferring a local checkpoint's bundle config."""
    if config is not None:
        return Path(config)

    bundle_config = checkpoint_path.parent / INFERENCE_CONFIG_FILENAME
    if bundle_config.is_file():
        click.echo(f"Using bundled inference config: {bundle_config}")
        return bundle_config

    package_dir = Path(__file__).resolve().parent
    candidates = [
        package_dir.parent / "configs" / "default.yaml",
        package_dir / "configs" / "default.yaml",
        Path("glare/configs/default.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            click.echo(f"Using default evaluation config: {candidate}")
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise click.ClickException(
        "No --config was provided, the local checkpoint has no sibling "
        f"{INFERENCE_CONFIG_FILENAME}, and the default evaluation config was not found. "
        f"Pass --config PATH or copy the complete inference bundle. Searched:\n  {searched}"
    )


def _resolve_optional_bundle_artifact(
    explicit_path: str | None,
    checkpoint_path: Path,
    filename: str,
    display_name: str,
    extra_dirs: list[Path] | None = None,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    allow_hf_bundle_artifact: bool = False,
) -> str | None:
    """Use an explicit path or a sibling bundle artifact if present."""
    if explicit_path:
        return explicit_path

    candidate_dirs = [checkpoint_path.parent]
    candidate_dirs.extend(extra_dirs or [])
    seen_dirs: set[Path] = set()
    for directory in candidate_dirs:
        try:
            key = directory.resolve()
        except OSError:
            key = directory
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        bundled_path = directory / filename
        if bundled_path.is_file():
            click.echo(f"Using bundled {display_name}: {bundled_path}")
            return str(bundled_path)

    if allow_hf_bundle_artifact and hf_repo_id:
        hf_artifact = _download_optional_hf_artifact(
            repo_id=hf_repo_id,
            filename=filename,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            display_name=display_name,
        )
        if hf_artifact is not None:
            return str(hf_artifact)
    return None


def _resolve_prediction_output_path(
    output_dir: Path,
    output_value: str | None,
    default_filename: str,
    option_name: str,
) -> Path:
    """Resolve a prediction output filename under the prediction output directory."""
    if output_value is None:
        return output_dir / default_filename

    requested = Path(output_value)
    if requested.is_absolute():
        raise click.ClickException(
            f"{option_name} must be a filename or relative path inside --output; "
            f"absolute paths are not allowed for prediction outputs."
        )

    resolved_output_dir = output_dir.resolve()
    output_path = output_dir / requested
    try:
        output_path.resolve().relative_to(resolved_output_dir)
    except ValueError as exc:
        raise click.ClickException(
            f"{option_name} must stay inside the --output directory."
        ) from exc

    return output_path


def _resolve_config_relative_model_ref(value: object, config_dir: Path) -> object:
    """Resolve local model/tokenizer references relative to the config file."""
    if value is None:
        return value
    text = str(value)
    if not text:
        return value
    path = Path(text)
    if path.is_absolute():
        return text

    candidate = config_dir / path
    is_explicit_relative = text.startswith(".") or text.startswith("..")
    if is_explicit_relative or candidate.exists():
        return str(candidate.resolve())
    return value


def _prepare_predict_config(cfg: dict, config_path: Path) -> dict:
    """Make bundle-relative prediction config paths independent of cwd."""
    config_dir = config_path.parent
    model_cfg = cfg.setdefault("model", {})
    current_base_model = model_cfg.get("base_model")
    resolved_base_model = _resolve_config_relative_model_ref(current_base_model, config_dir)
    if resolved_base_model != current_base_model:
        model_cfg["base_model"] = resolved_base_model
        click.echo(f"Resolved config-relative base model path: {resolved_base_model}")
        return cfg

    bundle_cfg = cfg.get("inference_bundle") or {}
    local_base_model = bundle_cfg.get("local_base_model_path")
    original_base_model = bundle_cfg.get("original_base_model")
    if local_base_model and current_base_model == original_base_model:
        candidate = _resolve_config_relative_model_ref(local_base_model, config_dir)
        if Path(str(candidate)).is_dir():
            model_cfg["base_model"] = candidate
            click.echo(f"Using bundled local base model path: {candidate}")
    return cfg


def _load_sibling_bundle_config(checkpoint_path: Path) -> tuple[Path, dict] | None:
    """Load bundle metadata adjacent to a local checkpoint, when present."""
    bundle_config_path = checkpoint_path.parent / INFERENCE_CONFIG_FILENAME
    if not bundle_config_path.is_file():
        return None
    try:
        with bundle_config_path.open() as handle:
            bundle_cfg = yaml.safe_load(handle)
    except Exception as exc:
        raise click.ClickException(
            f"Local checkpoint {checkpoint_path} has sibling {bundle_config_path}, "
            f"but that bundle config could not be read: {exc}"
        ) from exc
    if not isinstance(bundle_cfg, dict):
        raise click.ClickException(
            f"Local checkpoint {checkpoint_path} has sibling {bundle_config_path}, "
            "but that bundle config is not a YAML mapping."
        )
    return bundle_config_path, bundle_cfg


def _bundle_local_base_model(
    checkpoint_path: Path,
    bundle_config_path: Path,
    bundle_cfg: dict,
) -> tuple[Path, str | None, set[str]] | None:
    """Return and validate a sibling bundle's declared local base-model path."""
    bundle_model_cfg = bundle_cfg.get("model") or {}
    bundle_metadata = bundle_cfg.get("inference_bundle") or {}
    local_ref = bundle_metadata.get("local_base_model_path") or bundle_model_cfg.get(
        "base_model"
    )
    if local_ref is None:
        return None

    local_ref_text = str(local_ref)
    local_path = Path(local_ref_text)
    if not local_path.is_absolute():
        local_path = bundle_config_path.parent / local_path
    local_path = local_path.resolve()

    original_ref = bundle_metadata.get("original_base_model")
    matching_refs = {local_ref_text, str(local_path)}
    bundle_model_ref = bundle_model_cfg.get("base_model")
    if bundle_model_ref is not None:
        matching_refs.add(str(bundle_model_ref))
    if original_ref is not None:
        matching_refs.add(str(original_ref))

    return local_path, str(original_ref) if original_ref is not None else None, matching_refs


def _validate_bundle_local_base_model(
    checkpoint_path: Path,
    bundle_config_path: Path,
    local_path: Path,
) -> None:
    """Fail locally and actionably when a selected bundle is incomplete."""
    if not local_path.is_dir():
        raise click.ClickException(
            f"Local checkpoint {checkpoint_path} has sibling {bundle_config_path}, "
            "but its declared bundled base-model directory is missing: "
            f"{local_path}. Copy the complete bundle directory including "
            f"{local_path.name}/; do not copy only best_model.pt or the YAML file."
        )
    if not (local_path / "config.json").is_file():
        raise click.ClickException(
            f"Bundled base-model directory {local_path} exists but is incomplete: "
            "config.json is missing. Re-copy the complete inference bundle before retrying."
        )


def _prepare_config_for_local_checkpoint(
    cfg: dict,
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> dict:
    """Localize a matching base-model reference for a sibling inference bundle.

    Explicit configs retain their data, evaluation, and model hyperparameters.
    Only a base-model reference matching the sibling bundle's original or local
    reference is replaced, so a genuinely different explicit model is not masked.
    """
    if not isinstance(cfg, dict):
        raise click.ClickException("GLARE configuration must be a YAML mapping.")

    prepared = copy.deepcopy(cfg)
    if config_path is not None:
        prepared = _prepare_predict_config(prepared, config_path)

    sibling = _load_sibling_bundle_config(checkpoint_path)
    if sibling is None:
        return prepared

    bundle_config_path, bundle_cfg = sibling
    declared = _bundle_local_base_model(checkpoint_path, bundle_config_path, bundle_cfg)
    if declared is None:
        return prepared
    local_path, original_ref, matching_refs = declared

    model_cfg = prepared.setdefault("model", {})
    current_ref = model_cfg.get("base_model")
    if current_ref is None:
        raise click.ClickException(
            f"Config for local checkpoint {checkpoint_path} has no model.base_model. "
            f"Use its sibling config: {bundle_config_path}"
        )
    current_text = str(current_ref)
    try:
        current_resolved = str(Path(current_text).resolve())
    except OSError:
        current_resolved = current_text

    using_bundle_config = False
    if config_path is not None:
        try:
            using_bundle_config = config_path.resolve() == bundle_config_path.resolve()
        except OSError:
            pass
    matches_bundle = (
        using_bundle_config
        or current_text in matching_refs
        or current_resolved in matching_refs
        or (original_ref is not None and current_text == original_ref)
    )
    if matches_bundle:
        _validate_bundle_local_base_model(
            checkpoint_path,
            bundle_config_path,
            local_path,
        )
        local_text = str(local_path)
        if current_text != local_text:
            if original_ref is not None and current_text == original_ref:
                click.echo(
                    "Explicit config references the bundle's original Hub base model "
                    f"{original_ref!r}; using sibling local base model: {local_path}"
                )
            else:
                click.echo(f"Using bundled local base model path: {local_path}")
            model_cfg["base_model"] = local_text
    return prepared


def _resolve_predict_device_plan(device_value: str) -> list[torch.device]:
    """Resolve a validated single- or multi-device prediction plan."""
    normalized = device_value.lower().strip()
    if normalized == "auto":
        return [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    if normalized == "cpu":
        return [torch.device("cpu")]
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise click.ClickException(
                "--device cuda was requested, but CUDA is not available to PyTorch."
            )
        return [torch.device("cuda")]
    if normalized.startswith("cuda:"):
        selection = normalized.split(":", 1)[1]
        valid_list = bool(selection) and all(
            part.isdigit() for part in selection.split(",")
        )
        if selection != "all" and not valid_list:
            raise click.ClickException(
                "--device must be auto, cpu, cuda, cuda:all, or a "
                "comma-separated list such as cuda:0,2. Device indices must be "
                "non-negative integers with no empty entries or whitespace."
            )
        if not torch.cuda.is_available():
            raise click.ClickException(
                f"--device {normalized} was requested, but CUDA is not available to PyTorch."
            )
        device_count = torch.cuda.device_count()
        indices = list(range(device_count)) if selection == "all" else [
            int(part) for part in selection.split(",")
        ]
        if not indices:
            raise click.ClickException(
                f"--device {normalized} was requested, but no CUDA devices are visible."
            )
        if len(indices) != len(set(indices)):
            raise click.ClickException(
                f"--device {normalized} contains duplicate CUDA device indices."
            )
        out_of_range = [index for index in indices if index >= device_count]
        if out_of_range:
            raise click.ClickException(
                f"--device {normalized} requested logical CUDA device(s) "
                f"{out_of_range}, but only indices 0..{device_count - 1} are visible. "
                "Indices are interpreted after CUDA_VISIBLE_DEVICES filtering."
            )
        return [torch.device(f"cuda:{index}") for index in indices]
    raise click.ClickException(
        "--device must be auto, cpu, cuda, cuda:all, or a "
        "comma-separated list such as cuda:0,2."
    )


def _resolve_predict_device(device_value: str) -> torch.device:
    """Backward-compatible resolver for options selecting exactly one device."""
    plan = _resolve_predict_device_plan(device_value)
    if len(plan) != 1:
        raise click.ClickException(
            f"--device {device_value} selects multiple GPUs; use the multi-GPU launcher."
        )
    return plan[0]


def _balanced_contiguous_ranges(record_count: int, shard_count: int) -> list[tuple[int, int]]:
    """Return balanced, non-empty half-open ranges with no padding or repeats."""
    if record_count < 0 or shard_count < 1:
        raise ValueError("record_count must be non-negative and shard_count must be positive")
    active_shards = min(record_count, shard_count)
    if active_shards == 0:
        return []
    base, remainder = divmod(record_count, active_shards)
    ranges: list[tuple[int, int]] = []
    start = 0
    for shard_index in range(active_shards):
        size = base + (1 if shard_index < remainder else 0)
        stop = start + size
        ranges.append((start, stop))
        start = stop
    return ranges


def _resolve_predict_resources(
    cfg: dict,
    device: torch.device,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[int, int]:
    """Resolve effective prediction batch size and DataLoader worker count."""
    config_batch_size = int(cfg["training"]["batch_size"])
    cpu_count = _available_cpu_count()

    if batch_size is None:
        effective_batch_size = (
            config_batch_size if device.type == "cuda" else min(config_batch_size, 2)
        )
    else:
        effective_batch_size = batch_size

    if num_workers is None:
        effective_num_workers = (
            min(2, cpu_count) if device.type == "cuda" else min(1, cpu_count)
        )
    else:
        effective_num_workers = num_workers

    return effective_batch_size, effective_num_workers


def _available_cpu_count() -> int:
    """Return CPUs available to this process, honoring scheduler affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _predict_resource_warnings(
    *,
    gpu_count: int,
    num_workers: int,
    num_workers_explicit: bool,
    torch_threads: int | None,
) -> list[str]:
    """Describe explicit per-GPU settings that exceed available host CPUs."""
    available = _available_cpu_count()
    warnings: list[str] = []
    aggregate_workers = gpu_count * num_workers
    if num_workers_explicit and aggregate_workers > available:
        warnings.append(
            f"explicit --num-workers requests {aggregate_workers} DataLoader "
            f"workers across {gpu_count} GPUs, exceeding {available} available CPUs"
        )
    if torch_threads is not None:
        aggregate_threads = gpu_count * torch_threads
        if aggregate_threads > available:
            warnings.append(
                f"explicit --torch-threads requests {aggregate_threads} PyTorch "
                f"threads across {gpu_count} GPUs, exceeding {available} available CPUs"
            )
    return warnings


def _profile_echo(phase: str, seconds: float, details: str = "") -> None:
    """Emit one stable, monotonic profile line."""
    suffix = f"; {details}" if details else ""
    click.echo(f"PROFILE coordinator {phase}: {seconds:.3f}s{suffix}", err=True)


def _stage_path(source: Path, destination: Path) -> Path:
    """Atomically stage one local file, preferring a hard link when possible."""
    source = source.resolve()
    temporary = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def _stage_worker_artifacts(worker_config: dict, staging_dir: Path) -> dict:
    """Stage resolved worker artifacts once and return a repointed config copy."""
    staged = copy.deepcopy(worker_config)
    artifacts_dir = staging_dir / "artifacts"
    for key in ("checkpoint_path", "temperature_scaler_path", "mahal_detector_path"):
        value = staged.get(key)
        if not value:
            continue
        source = Path(value)
        destination = artifacts_dir / f"{key}-{source.name}"
        staged[key] = str(_stage_path(source, destination))

    base_model = staged.get("cfg", {}).get("model", {}).get("base_model")
    if base_model:
        source_dir = Path(str(base_model))
        source_parts = set(source_dir.parts)
        looks_like_hf_cache = (
            "snapshots" in source_parts
            and any(part.startswith("models--") for part in source_parts)
        )
        if source_dir.is_dir() and not looks_like_hf_cache:
            destination = artifacts_dir / "base-model"
            temporary = artifacts_dir / "base-model.partial"
            shutil.copytree(source_dir, temporary, copy_function=shutil.copy2)
            os.replace(temporary, destination)
            staged["cfg"]["model"]["base_model"] = str(destination)

    marker = artifacts_dir / ".complete"
    temporary_marker = artifacts_dir / ".complete.partial"
    temporary_marker.write_text("complete\n")
    os.replace(temporary_marker, marker)
    return staged


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True only for PyTorch CUDA out-of-memory exceptions."""
    oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
    return oom_error is not None and isinstance(exc, oom_error)


class _PredictCudaOOMError(RuntimeError):
    """Device-qualified CUDA OOM surfaced by prediction retry handling."""


def _load_prediction_runtime(
    cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
    temperature_scaler_path: Path | None,
    mahal_detector_path: Path | None,
    config_path: Path | None = None,
):
    """Construct one complete prediction runtime on the requested device."""
    from .data import get_tokenizer
    from .model import build_model
    from .ood import MahalanobisOOD, TemperatureScaler

    cfg = _prepare_config_for_local_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
    )
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict, rotary_cache_keys = _strip_rotary_embedding_cache_buffers(state_dict)
    if rotary_cache_keys:
        click.echo(
            f"Ignoring {len(rotary_cache_keys)} rotary embedding cache buffers "
            "from checkpoint; they will be recomputed.",
            err=True,
        )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    tokenizer = get_tokenizer(cfg["model"]["base_model"])
    temperature_scaler = (
        TemperatureScaler.from_file(temperature_scaler_path)
        if temperature_scaler_path else None
    )
    mahal_detector = (
        MahalanobisOOD.from_file(mahal_detector_path) if mahal_detector_path else None
    )
    return model, tokenizer, temperature_scaler, mahal_detector


def _predict_with_oom_backoff(
    *,
    model,
    tokenizer,
    temperature_scaler,
    mahal_detector,
    fasta_path: Path,
    device: torch.device,
    threshold: float,
    batch_size: int,
    batch_size_was_explicit: bool,
    num_workers: int,
    max_length: int,
    use_hierarchical_confidence: bool,
    mahal_threshold: float | None,
    label_space: str | dict,
    record_range: tuple[int, int] | None = None,
    include_input_index: bool = False,
    input_index_offset: int = 0,
    ood_method: str = "none",
    timing_metrics: dict | None = None,
    sliding_window_overlap: float | None = 0.5,
) -> tuple[object, int]:
    """Predict a complete range, retrying only automatic CUDA OOM failures."""
    from .predict import predict_fasta

    effective_batch_size = batch_size
    while True:
        try:
            frame = predict_fasta(
                model=model,
                fasta_path=fasta_path,
                tokenizer=tokenizer,
                device=device,
                threshold=threshold,
                batch_size=effective_batch_size,
                num_workers=num_workers,
                max_length=max_length,
                use_hierarchical_confidence=use_hierarchical_confidence,
                temperature_scaler=temperature_scaler,
                mahal_detector=mahal_detector,
                mahal_threshold=mahal_threshold,
                label_space=label_space,
                record_range=record_range,
                include_input_index=include_input_index,
                input_index_offset=input_index_offset,
                ood_method=ood_method,
                timing_metrics=timing_metrics,
                sliding_window_overlap=sliding_window_overlap,
            )
            return frame, effective_batch_size
        except Exception as exc:
            if not _is_cuda_oom(exc) or device.type != "cuda":
                raise
            device_label = str(device)
            if batch_size_was_explicit:
                raise _PredictCudaOOMError(
                    f"CUDA out of memory on {device_label} during prediction with "
                    f"explicit per-GPU --batch-size {effective_batch_size}. Retry "
                    "with a smaller per-GPU value."
                ) from exc
            if effective_batch_size <= 1:
                raise _PredictCudaOOMError(
                    f"CUDA out of memory on {device_label} during prediction even "
                    "with per-GPU batch size 1."
                ) from exc
            torch.cuda.empty_cache()
            next_batch_size = max(1, effective_batch_size // 2)
            click.echo(
                f"CUDA out of memory on {device_label}; retrying its complete "
                f"shard with per-GPU batch size {next_batch_size}.",
                err=True,
            )
            effective_batch_size = next_batch_size


def _predict_gpu_worker(
    worker_index: int,
    device_index: int,
    record_range: tuple[int, int],
    shard_path_text: str,
    status_queue,
    worker_config: dict,
) -> None:
    """Spawn-safe worker target for one contiguous prediction shard."""
    device = torch.device(f"cuda:{device_index}")
    shard_path = Path(shard_path_text)
    worker_started = time.monotonic()
    try:
        torch.cuda.set_device(device_index)
        torch_threads = worker_config.get("torch_threads")
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)

        model, tokenizer, temperature_scaler, mahal_detector = _load_prediction_runtime(
            worker_config["cfg"],
            Path(worker_config["checkpoint_path"]),
            device,
            Path(worker_config["temperature_scaler_path"])
            if worker_config.get("temperature_scaler_path") else None,
            Path(worker_config["mahal_detector_path"])
            if worker_config.get("mahal_detector_path") else None,
        )
        runtime_load_seconds = time.monotonic() - worker_started
        timing_metrics: dict = {}
        frame, final_batch_size = _predict_with_oom_backoff(
            model=model,
            tokenizer=tokenizer,
            temperature_scaler=temperature_scaler,
            mahal_detector=mahal_detector,
            fasta_path=Path(worker_config["input_path"]),
            device=device,
            threshold=worker_config["threshold"],
            batch_size=worker_config["batch_size"],
            batch_size_was_explicit=worker_config["batch_size_was_explicit"],
            num_workers=worker_config["num_workers"],
            max_length=worker_config["max_length"],
            use_hierarchical_confidence=worker_config["use_hierarchical_confidence"],
            mahal_threshold=worker_config["mahal_threshold"],
            label_space=worker_config["label_space"],
            record_range=None,
            include_input_index=True,
            input_index_offset=record_range[0],
            ood_method=worker_config.get("ood_method", "none"),
            timing_metrics=timing_metrics,
            sliding_window_overlap=worker_config.get("sliding_window_overlap", 0.5),
        )
        shard_write_started = time.monotonic()
        temporary_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        frame.to_pickle(temporary_path)
        os.replace(temporary_path, shard_path)
        shard_write_seconds = time.monotonic() - shard_write_started
        status_queue.put(
            {
                "status": "ok",
                "worker_index": worker_index,
                "device_index": device_index,
                "record_range": record_range,
                "batch_size": final_batch_size,
                "pid": os.getpid(),
                "runtime_load_seconds": runtime_load_seconds,
                "shard_write_seconds": shard_write_seconds,
                "ready_seconds": runtime_load_seconds,
                **timing_metrics,
            }
        )
    except BaseException as exc:
        status_queue.put(
            {
                "status": "error",
                "worker_index": worker_index,
                "device_index": device_index,
                "record_range": record_range,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _gather_prediction_shards(shard_paths: list[Path], record_count: int):
    """Gather, validate, source-sort, and de-privatize full-result shards."""
    import pandas as pd

    if record_count == 0:
        from .predict import empty_prediction_frame

        return empty_prediction_frame()
    frames = [pd.read_pickle(path) for path in shard_paths]
    gathered = pd.concat(frames, ignore_index=True)
    if "input_index" not in gathered.columns:
        raise RuntimeError("Prediction shard is missing required private input_index column.")
    if gathered["input_index"].isna().any():
        raise RuntimeError("Prediction shards contain missing input_index values.")
    try:
        indices = gathered["input_index"].astype(int)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Prediction shards contain non-integer input_index values.") from exc
    if indices.duplicated().any():
        duplicates = sorted(indices[indices.duplicated(keep=False)].unique().tolist())
        raise RuntimeError(f"Prediction shards contain duplicate input indices: {duplicates[:10]}")
    gathered = gathered.assign(input_index=indices).sort_values(
        "input_index", kind="mergesort"
    ).reset_index(drop=True)
    observed = gathered["input_index"].tolist()
    expected = list(range(record_count))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise RuntimeError(
            "Prediction shards are incomplete or out of range: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )
    return gathered.drop(columns=["input_index"])


def _launch_multi_gpu_prediction(
    devices: list[torch.device],
    ranges: list[tuple[int, int]],
    record_count: int,
    worker_config: dict,
    profile: bool = False,
    artifact_staging_dir: Path | None = None,
    record_offsets: list[int] | None = None,
):
    """Launch spawn workers, fail-fast as a group, and gather their shards."""
    context = torch.multiprocessing.get_context("spawn")
    run_started = time.monotonic()
    temp_parent = str(artifact_staging_dir) if artifact_staging_dir else None
    if artifact_staging_dir is not None:
        artifact_staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="glare-predict-", dir=temp_parent
    ) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        from .predict import prepare_sequence_shards

        shard_prepare_started = time.monotonic()
        input_shards = prepare_sequence_shards(
            Path(worker_config["input_path"]),
            ranges,
            temp_dir / "inputs",
            record_offsets=record_offsets,
        )
        if profile:
            _profile_echo(
                "record scan/shard preparation",
                time.monotonic() - shard_prepare_started,
                f"records={record_count}, shards={len(input_shards)}",
            )
        if artifact_staging_dir is not None:
            staging_started = time.monotonic()
            worker_config = _stage_worker_artifacts(worker_config, temp_dir)
            if profile:
                _profile_echo(
                    "artifact staging", time.monotonic() - staging_started
                )
        status_queue = context.Queue()
        processes = []
        shard_paths = []
        for worker_index, (device, record_range) in enumerate(zip(devices, ranges)):
            device_index = device.index
            if device_index is None:
                raise RuntimeError("Multi-GPU prediction requires indexed CUDA devices.")
            shard_path = temp_dir / f"shard-{worker_index:04d}.pkl"
            shard_paths.append(shard_path)
            per_worker_config = dict(worker_config)
            per_worker_config["input_path"] = str(input_shards[worker_index])
            process = context.Process(
                target=_predict_gpu_worker,
                args=(
                    worker_index,
                    device_index,
                    record_range,
                    str(shard_path),
                    status_queue,
                    per_worker_config,
                ),
            )
            process.start()
            processes.append(process)

        statuses: dict[int, dict] = {}
        failure: dict | None = None
        while len(statuses) < len(processes) and failure is None:
            try:
                status = status_queue.get(timeout=0.1)
            except queue.Empty:
                for worker_index, process in enumerate(processes):
                    if process.exitcode not in (None, 0) and worker_index not in statuses:
                        failure = {
                            "device_index": devices[worker_index].index,
                            "record_range": ranges[worker_index],
                            "error": f"worker exited with code {process.exitcode}",
                        }
                        break
                if all(not process.is_alive() for process in processes) and failure is None:
                    missing_status = sorted(set(range(len(processes))) - set(statuses))
                    if missing_status:
                        worker_index = missing_status[0]
                        failure = {
                            "device_index": devices[worker_index].index,
                            "record_range": ranges[worker_index],
                            "error": "worker exited without reporting status",
                        }
                continue
            statuses[status["worker_index"]] = status
            if status["status"] == "error":
                failure = status

        if failure is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise click.ClickException(
                "Multi-GPU prediction failed on logical "
                f"cuda:{failure.get('device_index')} for record range "
                f"{tuple(failure.get('record_range', ()))}: "
                f"{failure.get('error', 'unknown worker error')}"
            )

        for process in processes:
            process.join()
        bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
        if bad_exits:
            raise click.ClickException(
                f"Multi-GPU prediction worker(s) exited unsuccessfully: {bad_exits}"
            )
        for worker_index in sorted(statuses):
            status = statuses[worker_index]
            if profile:
                click.echo(
                    "PROFILE worker "
                    f"device=cuda:{status['device_index']} pid={status['pid']} "
                    f"records={status.get('record_count', 0)} "
                    f"batches={status.get('batch_count', 0)} "
                    f"final_batch_size={status.get('final_batch_size', 0)} "
                    f"runtime_load={status['runtime_load_seconds']:.3f}s "
                    f"input_setup={status.get('input_setup_seconds', 0.0):.3f}s "
                    f"first_batch={status.get('first_batch_seconds', 0.0):.3f}s "
                    f"inference={status.get('inference_seconds', 0.0):.3f}s "
                    f"shard_write={status['shard_write_seconds']:.3f}s",
                    err=True,
                )
        if profile:
            ready = max(status.get("ready_seconds", 0.0) for status in statuses.values())
            _profile_echo("worker launch-to-ready", ready)
            _profile_echo("inference completion", time.monotonic() - run_started)
        gather_started = time.monotonic()
        gathered = _gather_prediction_shards(shard_paths, record_count)
        if profile:
            _profile_echo("shard gather", time.monotonic() - gather_started)
        return gathered


def _ensure_hf_bundle_snapshot(
    cfg: dict,
    config_path: Path,
    hf_repo_id: str,
    hf_revision: str | None,
    hf_cache_dir: str | None,
) -> None:
    """Populate local bundle directories from a Hugging Face repo snapshot."""
    model_cfg = cfg.get("model") or {}
    base_model = model_cfg.get("base_model")
    bundle_cfg = cfg.get("inference_bundle") or {}
    local_base_model = bundle_cfg.get("local_base_model_path")
    local_ref = local_base_model or base_model
    if local_ref is None:
        return

    local_ref_text = str(local_ref)
    local_ref_path = Path(local_ref_text)
    if local_ref_path.is_absolute():
        return
    is_bundle_relative = local_ref_text.startswith(".") or local_base_model is not None
    if not is_bundle_relative:
        return
    if (config_path.parent / local_ref_path).exists():
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return

    try:
        snapshot_dir = snapshot_download(
            repo_id=hf_repo_id,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            allow_patterns=[
                DEFAULT_HF_CHECKPOINT_FILENAME,
                INFERENCE_CONFIG_FILENAME,
                LABELS_FILENAME,
                "manifest.json",
                TEMPERATURE_SCALER_FILENAME,
                MAHAL_OOD_FILENAME,
                "base_model_*/*",
                "base_model_*/**/*",
            ],
        )
    except Exception as exc:
        click.echo(
            "[WARN] Could not populate Hugging Face inference-bundle snapshot; "
            f"local bundle base-model path may be unavailable: {exc}",
            err=True,
        )
        return

    click.echo(f"Using Hugging Face inference bundle snapshot: {snapshot_dir}")


def _candidate_bundle_label_paths(cfg: dict, config_path: Path) -> list[Path]:
    """Return possible labels.json paths next to a bundle config."""
    config_dir = config_path.parent
    candidates: list[Path] = []
    bundle_cfg = cfg.get("inference_bundle") or {}
    labels_filename = bundle_cfg.get("labels_filename")
    if labels_filename:
        labels_path = Path(str(labels_filename))
        candidates.append(labels_path if labels_path.is_absolute() else config_dir / labels_path)
    candidates.append(config_dir / LABELS_FILENAME)
    return candidates


def _load_bundle_label_metadata(cfg: dict, config_path: Path) -> dict | None:
    """Load bundle label metadata for prediction decoding when available."""
    from .predict import load_label_space_metadata

    seen: set[Path] = set()
    for labels_path in _candidate_bundle_label_paths(cfg, config_path):
        try:
            key = labels_path.resolve()
        except OSError:
            key = labels_path
        if key in seen:
            continue
        seen.add(key)
        if labels_path.is_file():
            metadata = load_label_space_metadata(labels_path)
            click.echo(f"Using bundled label metadata: {labels_path}")
            return metadata
    return None

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """GLARE: Genomic Language model for Annotation of Repetitive Elements."""


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--config", "-c",
    default="glare/configs/default.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.option(
    "--output-dir", "-o",
    default="glare_output",
    show_default=True,
    help="Directory to write checkpoints and training log.",
)
@click.option(
    "--data-dir",
    default=None,
    help="Override data.repbase_dir from config (legacy Repbase mode).",
)
@click.option(
    "--repbase-label-space",
    type=click.Choice(["legacy45", "panteon30"], case_sensitive=False),
    default=None,
    help="Repbase training label space: legacy 45 labels or PanTEon/GLARE 30 labels.",
)
@click.option(
    "--fasta-path",
    default=None,
    help="Override data.fasta_path from config (PanTEon mode).",
)
@click.option(
    "--taxonomy-metadata-csv",
    default=None,
    help="Enable taxonomy holdout with this metadata CSV path.",
)
@click.option(
    "--taxonomy-filter-column",
    default=None,
    help="Metadata column to filter for taxonomy holdout, e.g. Order.",
)
@click.option(
    "--taxonomy-filter-value",
    default=None,
    help="Metadata value to hold out, e.g. Poales.",
)
@click.option(
    "--taxonomy-species-column",
    default=None,
    help="Metadata species column for taxonomy holdout, e.g. Species.",
)
def train(
    config: str,
    output_dir: str,
    data_dir: str | None,
    repbase_label_space: str | None,
    fasta_path: str | None,
    taxonomy_metadata_csv: str | None,
    taxonomy_filter_column: str | None,
    taxonomy_filter_value: str | None,
    taxonomy_species_column: str | None,
) -> None:
    """Train GLARE model (single-GPU; use train_ddp.py for multi-GPU).

    PanTEon mode is activated when --fasta-path is provided or when the config
    contains ``data.fasta_path``.  Legacy Repbase mode is used otherwise.
    """
    import os
    from .train import train as _train

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    if fasta_path is not None:
        cfg.setdefault("data", {})["fasta_path"] = fasta_path
    elif data_dir is not None:
        cfg.setdefault("data", {})["repbase_dir"] = data_dir
    if repbase_label_space is not None:
        cfg.setdefault("data", {})["repbase_label_space"] = repbase_label_space.lower()

    taxonomy_cli_values = {
        "metadata_csv": taxonomy_metadata_csv,
        "filter_column": taxonomy_filter_column,
        "filter_value": taxonomy_filter_value,
        "species_column": taxonomy_species_column,
    }
    if any(value is not None for value in taxonomy_cli_values.values()):
        missing = [key for key, value in taxonomy_cli_values.items() if value is None]
        if missing:
            raise click.UsageError(
                "Taxonomy holdout CLI overrides require all of: "
                "--taxonomy-metadata-csv, --taxonomy-filter-column, "
                "--taxonomy-filter-value, --taxonomy-species-column. "
                f"Missing: {', '.join(missing)}"
            )
        holdout_cfg = cfg.setdefault("data", {}).setdefault("taxonomy_holdout", {})
        holdout_cfg.update({"enabled": True, **taxonomy_cli_values})

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _train(config=cfg, output_dir=Path(output_dir), local_rank=local_rank)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--checkpoint", "-ckpt",
    required=True,
    help="Path to model checkpoint .pt file.",
)
@click.option(
    "--config", "-c",
    default=None,
    help=(
        "Path to YAML config file. If omitted for a local bundled checkpoint, "
        "uses its sibling inference_config.yaml."
    ),
)
@click.option(
    "--output-dir", "-o",
    default="glare_results",
    show_default=True,
    help="Directory to save evaluation CSV.",
)
@click.option(
    "--panteon",
    is_flag=True,
    default=False,
    help="Use PanTEon evaluation protocol (no threshold, weighted F1 on test split).",
)
@click.option(
    "--external-repbase",
    is_flag=True,
    default=False,
    help="Evaluate an external Repbase FASTA/REF after mapping into the PanTEon 30-label space.",
)
@click.option(
    "--fasta-path",
    default=None,
    help="Path to PanTEon FASTA file (required when --panteon is set).",
)
@click.option(
    "--repbase-fasta",
    default=None,
    help="Path to external Repbase FASTA/REF file or directory of .ref files (required with --external-repbase).",
)
@click.option(
    "--toml-path",
    default=None,
    help="Path to Repbase→RepeatMasker TOML mapping (defaults to data.toml_mapping or glare/data/repbase-to-repeatmasker.toml).",
)
@click.option(
    "--taxonomy-metadata-csv",
    default=None,
    help="Evaluate PanTEon using taxonomy holdout from this metadata CSV path.",
)
@click.option(
    "--taxonomy-filter-column",
    default=None,
    help="Metadata column to filter for taxonomy holdout, e.g. Order.",
)
@click.option(
    "--taxonomy-filter-value",
    default=None,
    help="Metadata value to hold out, e.g. Poales.",
)
@click.option(
    "--taxonomy-species-column",
    default=None,
    help="Metadata species column for taxonomy holdout, e.g. Species.",
)
@click.option(
    "--temperature-scaler",
    default=None,
    help="Path to temperature_scaler.json for post-hoc calibration.",
)
@click.option(
    "--mahal-detector",
    default=None,
    help="Path to mahal_ood.npz for Mahalanobis OOD scoring.",
)
@click.option(
    "--ood-method",
    type=click.Choice(["hier_conf", "energy", "mahal", "combined"], case_sensitive=False),
    default="hier_conf",
    show_default=True,
    help="OOD method used for the 'classified' flag in legacy Repbase evaluation.",
)
@click.option(
    "--ood-threshold",
    default=None,
    type=float,
    help="Override confidence threshold for OOD method (default: 0.7 for hier_conf).",
)
def evaluate(
    checkpoint: str,
    config: str | None,
    output_dir: str,
    panteon: bool,
    external_repbase: bool,
    fasta_path: str | None,
    repbase_fasta: str | None,
    toml_path: str | None,
    taxonomy_metadata_csv: str | None,
    taxonomy_filter_column: str | None,
    taxonomy_filter_value: str | None,
    taxonomy_species_column: str | None,
    temperature_scaler: str | None,
    mahal_detector: str | None,
    ood_method: str,
    ood_threshold: float | None,
) -> None:
    """Evaluate GLARE on the holdout benchmark.

    Use ``--panteon`` for the PanTEon benchmark (requires ``--fasta-path``).
    Use ``--external-repbase`` for external Repbase FASTA/REF files mapped into
    the PanTEon 30-superfamily label space.
    Without ``--panteon``, evaluates on the legacy Repbase holdout groups.
    """
    from .data import get_tokenizer
    from .model import build_model
    from .ood import TemperatureScaler, MahalanobisOOD

    checkpoint_path = Path(checkpoint)
    config_path = _resolve_evaluate_config_for_checkpoint(config, checkpoint_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = _prepare_config_for_local_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
    )

    taxonomy_cli_values = {
        "metadata_csv": taxonomy_metadata_csv,
        "filter_column": taxonomy_filter_column,
        "filter_value": taxonomy_filter_value,
        "species_column": taxonomy_species_column,
    }
    if any(value is not None for value in taxonomy_cli_values.values()):
        missing = [key for key, value in taxonomy_cli_values.items() if value is None]
        if missing:
            raise click.UsageError(
                "Taxonomy holdout CLI overrides require all of: "
                "--taxonomy-metadata-csv, --taxonomy-filter-column, "
                "--taxonomy-filter-value, --taxonomy-species-column. "
                f"Missing: {', '.join(missing)}"
            )
        holdout_cfg = cfg.setdefault("data", {}).setdefault("taxonomy_holdout", {})
        holdout_cfg.update({"enabled": True, **taxonomy_cli_values})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Device: {device}")

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        click.echo(
            f"[WARN] Missing keys in checkpoint: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}",
            err=True,
        )
    rotary_keys = [k for k in unexpected if "rotary_embedding" in k]
    other_unexpected = [k for k in unexpected if "rotary_embedding" not in k]
    if rotary_keys:
        click.echo(
            f"[INFO] Ignoring {len(rotary_keys)} rotary_embedding cache buffers "
            "(will be recomputed).",
            err=True,
        )
    if other_unexpected:
        click.echo(
            f"[WARN] Unexpected non-rotary keys: {other_unexpected[:5]}"
            f"{'...' if len(other_unexpected) > 5 else ''}",
            err=True,
        )
    model.eval()
    model.to(device)

    # Tokenizer
    tokenizer = get_tokenizer(cfg["model"]["base_model"])

    # OOD detectors (optional)
    ts = TemperatureScaler.from_file(temperature_scaler) if temperature_scaler else None
    md = MahalanobisOOD.from_file(mahal_detector) if mahal_detector else None
    effective_threshold = ood_threshold if ood_threshold is not None else 0.7

    if panteon and external_repbase:
        raise click.UsageError("Use only one of --panteon or --external-repbase.")

    if external_repbase:
        # ── External Repbase → PanTEon-label evaluation ──────────────────────
        from .evaluate import evaluate_external_repbase_full

        if repbase_fasta is None:
            raise click.UsageError("--external-repbase requires --repbase-fasta.")
        resolved_toml = (
            toml_path
            or cfg.get("data", {}).get("toml_mapping")
            or "glare/data/repbase-to-repeatmasker.toml"
        )

        summary_df, per_class_df = evaluate_external_repbase_full(
            model=model,
            fasta_path=Path(repbase_fasta),
            toml_path=Path(resolved_toml),
            tokenizer=tokenizer,
            device=device,
            config=cfg,
            output_dir=Path(output_dir),
        )

        click.echo("\nExternal Repbase summary:")
        click.echo(summary_df.to_string(index=False))
        click.echo("\nExternal Repbase per-class F1:")
        click.echo(per_class_df.to_string(index=False))

    elif panteon:
        # ── PanTEon evaluation ───────────────────────────────────────────────
        from .evaluate import evaluate_panteon_full

        # Resolve fasta_path: CLI flag > config
        resolved_fasta = fasta_path or cfg.get("data", {}).get("fasta_path")
        if resolved_fasta is None:
            raise click.UsageError(
                "--panteon requires --fasta-path or data.fasta_path in config."
            )

        results = evaluate_panteon_full(
            model=model,
            fasta_path=Path(resolved_fasta),
            tokenizer=tokenizer,
            device=device,
            config=cfg,
            output_dir=Path(output_dir),
        )

        click.echo("\nPer-class F1 (superfamily):")
        click.echo(results.to_string(index=False))

    else:
        # ── Legacy Repbase holdout evaluation ────────────────────────────────
        from .evaluate import evaluate_holdout_groups

        holdout_dir = Path(cfg["data"]["holdout_dir"])
        resolved_toml_path = Path(cfg["data"]["toml_mapping"])
        thresholds = cfg["evaluation"]["thresholds"]

        results = evaluate_holdout_groups(
            model=model,
            holdout_dir=holdout_dir,
            toml_path=resolved_toml_path,
            tokenizer=tokenizer,
            device=device,
            config=cfg,
            thresholds=thresholds,
            output_dir=Path(output_dir),
            temperature_scaler=ts,
            mahal_detector=md,
            ood_method=ood_method,
        )

        click.echo("\nResults:")
        click.echo(results.to_string(index=False))


@main.command("inspect-repbase30")
@click.option(
    "--data-dir",
    default="glare/data/repbase31_filtered",
    show_default=True,
    help="Train/validation directory containing Repbase .ref files to parse/map.",
)
@click.option(
    "--holdout-dir",
    default="glare/data/repbase31_holdout",
    show_default=True,
    help="External test/holdout directory containing Repbase .ref files to parse/map.",
)
@click.option(
    "--toml-path",
    default="glare/data/repbase-to-repeatmasker.toml",
    show_default=True,
    help="Repbase category → RepeatMasker TOML mapping.",
)
@click.option("--val-fraction", default=0.1, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
def inspect_repbase30(
    data_dir: str,
    holdout_dir: str,
    toml_path: str,
    val_fraction: float,
    seed: int,
) -> None:
    """Parse Repbase REF files and summarize PanTEon30 mapping without training."""
    from .constants import get_label_space
    from .data import load_repbase_dir_for_panteon_training, split_panteon_train_val_records

    train_val_records, diagnostics = load_repbase_dir_for_panteon_training(
        data_dir=Path(data_dir),
        toml_path=Path(toml_path),
    )
    holdout_records, holdout_diagnostics = load_repbase_dir_for_panteon_training(
        data_dir=Path(holdout_dir),
        toml_path=Path(toml_path),
    )
    label_space = get_label_space("panteon30")
    train_recs, val_recs, split_stats = split_panteon_train_val_records(
        records=train_val_records,
        val_fraction=val_fraction,
        seed=seed,
        superfamily2idx=label_space["superfamily2idx"],
    )
    click.echo(
        "Repbase→PanTEon30 train/validation parse/mapping summary: "
        f"loaded={diagnostics['loaded_records']}, "
        f"mapped={diagnostics['mapped_records']}, "
        f"skipped={diagnostics['skipped_records']}, "
        f"ref_files={diagnostics['ref_files']}"
    )
    click.echo("\nTrain/validation skipped reasons:")
    for label, count in diagnostics["skipped_reason_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nTrain/validation mapped order counts:")
    for label, count in diagnostics["panteon_order_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nTrain/validation mapped superfamily counts:")
    for label, count in diagnostics["panteon_superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo(
        "\nRepbase→PanTEon30 external test parse/mapping summary: "
        f"loaded={holdout_diagnostics['loaded_records']}, "
        f"mapped={holdout_diagnostics['mapped_records']}, "
        f"skipped={holdout_diagnostics['skipped_records']}, "
        f"ref_files={holdout_diagnostics['ref_files']}"
    )
    click.echo("\nExternal test skipped reasons:")
    for label, count in holdout_diagnostics["skipped_reason_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nExternal test mapped order counts:")
    for label, count in holdout_diagnostics["panteon_order_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nExternal test mapped superfamily counts:")
    for label, count in holdout_diagnostics["panteon_superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo(
        "\nFinal split sizes: "
        f"train={len(train_recs)}, val={len(val_recs)}, external_test={len(holdout_records)} "
        f"(split_mode={split_stats['split_mode']})"
    )


@main.command("inspect-shared-labels")
@click.option(
    "--config", "config_path",
    default=None,
    help="Optional YAML config containing data.label_subset settings.",
)
@click.option(
    "--repbase-train-dir",
    default="glare/data/repbase31_filtered",
    show_default=True,
    help="Repbase train source directory used to compute mapped train labels.",
)
@click.option(
    "--repbase-holdout-dir",
    default="glare/data/repbase31_holdout",
    show_default=True,
    help="Optional Repbase external holdout directory for expected filtered counts.",
)
@click.option(
    "--repbase-toml-path",
    default="glare/data/repbase-to-repeatmasker.toml",
    show_default=True,
    help="Repbase category → RepeatMasker TOML mapping.",
)
@click.option(
    "--panteon-fasta-path",
    default="glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta",
    show_default=True,
    help="PanTEon FASTA used to compute train-side and holdout labels.",
)
@click.option(
    "--metadata-csv",
    default="glare/data/PanTEon_Database_metadata_v1.6.1.csv",
    show_default=True,
    help="PanTEon metadata CSV for taxonomy holdout.",
)
@click.option("--filter-column", default="Order", show_default=True)
@click.option("--filter-value", default="Poales", show_default=True)
@click.option("--species-column", default="Species", show_default=True)
@click.option("--val-fraction", default=0.1, show_default=True, type=float)
@click.option("--seed", default=7, show_default=True, type=int)
def inspect_shared_labels(
    config_path: str | None,
    repbase_train_dir: str,
    repbase_holdout_dir: str,
    repbase_toml_path: str,
    panteon_fasta_path: str,
    metadata_csv: str,
    filter_column: str,
    filter_value: str,
    species_column: str,
    val_fraction: float,
    seed: int,
) -> None:
    """Inspect conservative Repbase↔PanTEon shared training labels."""
    from .constants import get_label_space
    from .data import (
        compute_train_intersection_label_subset,
        filter_records_by_superfamilies,
        load_panteon_dataset,
        load_repbase_dir_for_panteon_training,
        partition_panteon_records_by_taxonomy_holdout,
        split_panteon_records,
        split_panteon_train_val_records,
    )

    if config_path is not None:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = {"data": {}}

    data_cfg = cfg.setdefault("data", {})
    taxonomy_holdout = data_cfg.get("taxonomy_holdout") or {
        "enabled": True,
        "metadata_csv": metadata_csv,
        "filter_column": filter_column,
        "filter_value": filter_value,
        "species_column": species_column,
        "filter_match": "exact",
        "fasta_species_mode": "after_at",
        "species_match": "normalized_exact",
        "fail_on_no_matches": True,
    }
    label_subset_cfg = data_cfg.setdefault("label_subset", {})
    label_subset_cfg.update({
        "enabled": True,
        "mode": "train_intersection",
        "repbase_train_dir": label_subset_cfg.get("repbase_train_dir", repbase_train_dir),
        "repbase_toml_path": label_subset_cfg.get("repbase_toml_path", repbase_toml_path),
        "panteon_fasta_path": label_subset_cfg.get("panteon_fasta_path", panteon_fasta_path),
        "panteon_taxonomy_holdout": label_subset_cfg.get(
            "panteon_taxonomy_holdout",
            taxonomy_holdout,
        ),
    })

    labels, stats = compute_train_intersection_label_subset(label_subset_cfg, data_cfg)
    click.echo("Repbase train labels/counts:")
    for label, count in stats["repbase_train_superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nPanTEon train-side labels/counts after taxonomy holdout:")
    for label, count in stats["panteon_train_side_superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nShared-label intersection:")
    click.echo("  " + ", ".join(labels))
    click.echo(f"  n_labels={len(labels)}")
    click.echo("\nExcluded labels:")
    click.echo(
        "  Present in Repbase train but absent from PanTEon train-side: "
        + (", ".join(stats["repbase_only_labels_excluded"]) or "None")
    )
    click.echo(
        "  Present in PanTEon train-side but absent from Repbase train: "
        + (", ".join(stats["panteon_only_labels_excluded"]) or "None")
    )

    label_space = get_label_space("panteon30")
    repbase_train_records, _ = load_repbase_dir_for_panteon_training(
        Path(repbase_train_dir), Path(repbase_toml_path)
    )
    repbase_train_records, repbase_train_filter = filter_records_by_superfamilies(
        repbase_train_records,
        labels,
        dataset_name="Repbase train source inspection",
        log=False,
    )
    rep_train, rep_val, _rep_split_stats = split_panteon_train_val_records(
        records=repbase_train_records,
        val_fraction=val_fraction,
        seed=seed,
        superfamily2idx=label_space["superfamily2idx"],
    )

    repbase_holdout_records, _ = load_repbase_dir_for_panteon_training(
        Path(repbase_holdout_dir), Path(repbase_toml_path)
    )
    repbase_holdout_records, repbase_holdout_filter = filter_records_by_superfamilies(
        repbase_holdout_records,
        labels,
        dataset_name="Repbase external holdout inspection",
        log=False,
    )

    panteon_records = load_panteon_dataset(Path(panteon_fasta_path))
    panteon_records, panteon_all_filter = filter_records_by_superfamilies(
        panteon_records,
        labels,
        dataset_name="PanTEon all records inspection",
        log=False,
    )
    pan_train, pan_val, pan_test, pan_split_stats = split_panteon_records(
        records=panteon_records,
        val_fraction=val_fraction,
        seed=seed,
        taxonomy_holdout=taxonomy_holdout,
    )
    pan_train_side, pan_holdout, _ = partition_panteon_records_by_taxonomy_holdout(
        panteon_records,
        taxonomy_holdout=taxonomy_holdout,
    )

    click.echo("\nExpected record counts after shared-label filtering:")
    click.echo(
        "  Repbase train source: "
        f"kept={repbase_train_filter['kept_records']}/"
        f"{repbase_train_filter['total_records']} skipped={repbase_train_filter['skipped_records']}"
    )
    click.echo(
        f"  Repbase train/val split: train={len(rep_train)}, val={len(rep_val)}"
    )
    click.echo(
        "  Repbase external holdout: "
        f"kept={repbase_holdout_filter['kept_records']}/"
        f"{repbase_holdout_filter['total_records']} skipped={repbase_holdout_filter['skipped_records']}"
    )
    click.echo(
        "  PanTEon all records: "
        f"kept={panteon_all_filter['kept_records']}/"
        f"{panteon_all_filter['total_records']} skipped={panteon_all_filter['skipped_records']}"
    )
    click.echo(
        "  PanTEon taxonomy sides after filtering: "
        f"train_side={len(pan_train_side)}, holdout_test_side={len(pan_holdout)}"
    )
    click.echo(
        "  PanTEon train/val/test split after filtering: "
        f"train={len(pan_train)}, val={len(pan_val)}, test={len(pan_test)} "
        f"(split_mode={pan_split_stats['split_mode']})"
    )


@main.command("inspect-combined")
@click.option(
    "--config", "config_path",
    default="glare/configs/combined_repbase_panteon_shared_labels.yaml",
    show_default=True,
    help="YAML config containing combined Repbase+PanTEon data settings.",
)
def inspect_combined(config_path: str) -> None:
    """Parse combined Repbase+PanTEon sources and summarize counts only."""
    from .constants import get_label_space
    from .data import (
        build_combined_repbase_panteon_records,
        count_record_labels,
        format_split_stats,
        resolve_superfamily_label_subset,
    )

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})
    labels, label_stats = resolve_superfamily_label_subset(data_cfg)
    label_space = get_label_space(str(data_cfg.get("label_space", "panteon30")))

    train_recs, val_recs, test_sets, diagnostics = build_combined_repbase_panteon_records(
        data_config=data_cfg,
        val_fraction=float(training_cfg.get("val_fraction", 0.1)),
        seed=int(training_cfg.get("seed", 7)),
        superfamily2idx=label_space["superfamily2idx"],
        label_subset_labels=labels,
        log=False,
    )

    if labels is None:
        label_space_labels = list(label_space["superfamily_labels"])
        click.echo("Combined Repbase+PanTEon full-label-space inspection")
        click.echo(
            f"Label space {label_space['name']} ({len(label_space_labels)}): "
            f"{', '.join(label_space_labels)}"
        )
    else:
        click.echo("Combined Repbase+PanTEon label-subset inspection")
        click.echo(f"Subset labels ({len(labels)}): {', '.join(labels)}")

    if label_stats:
        click.echo(
            "Excluded labels: "
            f"Repbase-only={label_stats.get('repbase_only_labels_excluded', [])}; "
            f"PanTEon-only={label_stats.get('panteon_only_labels_excluded', [])}"
        )

    click.echo("\nParse/filter counts:")
    click.echo(
        "  Repbase train source: "
        f"loaded={diagnostics['repbase_train_diagnostics']['loaded_records']} "
        f"mapped={diagnostics['repbase_train_diagnostics']['mapped_records']} "
        f"kept={diagnostics['repbase_train_records_kept']}"
    )
    click.echo(
        "  PanTEon train side: "
        f"total={diagnostics['panteon_total_records']} "
        f"partition_train_side={diagnostics['panteon_partition_stats'].get('train_side_records')} "
        f"kept={diagnostics['panteon_train_side_records_kept']}"
    )
    panteon_test_name = diagnostics.get("panteon_test_name", "panteon_poales")
    repbase_test_name = diagnostics.get("repbase_test_name", "repbase_grass")
    click.echo(
        f"  PanTEon external test [{panteon_test_name}]: "
        f"partition_holdout={diagnostics['panteon_partition_stats'].get('holdout_records')} "
        f"kept={diagnostics.get('panteon_test_records_kept', diagnostics['panteon_poales_test_records_kept'])}"
    )
    click.echo(
        f"  Repbase external test [{repbase_test_name}]: "
        f"loaded={diagnostics['repbase_test_diagnostics']['loaded_records']} "
        f"mapped={diagnostics['repbase_test_diagnostics']['mapped_records']} "
        f"kept={diagnostics.get('repbase_test_records_kept', diagnostics['repbase_grass_test_records_kept'])}"
    )

    click.echo("\nCombined split:")
    click.echo(format_split_stats(diagnostics["split_stats"]))
    click.echo(
        "Source counts: "
        f"combined={diagnostics['combined_source_counts']} "
        f"train={diagnostics['train_source_counts']} "
        f"val={diagnostics['val_source_counts']}"
    )
    click.echo(
        f"Final sizes: train={len(train_recs)}, val={len(val_recs)}, "
        f"{panteon_test_name}={len(test_sets[panteon_test_name])}, "
        f"{repbase_test_name}={len(test_sets[repbase_test_name])}"
    )

    click.echo("\nTrain superfamily counts:")
    for label, count in count_record_labels(train_recs)["superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo("\nValidation superfamily counts:")
    for label, count in count_record_labels(val_recs)["superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo(f"\nPanTEon external test superfamily counts [{panteon_test_name}]:")
    for label, count in count_record_labels(test_sets[panteon_test_name])["superfamily_counts"].items():
        click.echo(f"  {label}: {count}")
    click.echo(f"\nRepbase external test superfamily counts [{repbase_test_name}]:")
    for label, count in count_record_labels(test_sets[repbase_test_name])["superfamily_counts"].items():
        click.echo(f"  {label}: {count}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--checkpoint", "-ckpt",
    required=False,
    default=None,
    hidden=True,
    help=(
        "Path to local model checkpoint .pt file. If omitted, GLARE downloads "
        "the default checkpoint from Hugging Face."
    ),
)
@click.option(
    "--config", "-c",
    default=None,
    hidden=True,
    show_default=DEFAULT_PREDICT_CONFIG_DISPLAY,
    help="Path to YAML config file.",
)
@click.option(
    "--input", "-i",
    required=True,
    help="Input FASTA or .ref file.",
)
@click.option(
    "--output", "-o",
    required=True,
    help="Output directory for all prediction files.",
)
@click.option(
    "--summary-output",
    default=None,
    hidden=True,
    help=(
        "Class-summary CSV filename or relative path inside --output. "
        "Defaults to 'class_summary.csv'."
    ),
)
@click.option(
    "--figure-output",
    default=None,
    hidden=True,
    help=(
        "Class-composition figure filename or relative path inside --output. "
        "Defaults to 'class_composition.png'."
    ),
)
@click.option(
    "--stacked-figure-output",
    default=None,
    hidden=True,
    help=(
        "Stacked class-composition figure filename or relative path inside --output. "
        "Defaults to 'stacked_class_composition.png'."
    ),
)
@click.option(
    "--fasta-output",
    default=None,
    hidden=True,
    help=(
        "Annotated FASTA filename or relative path inside --output with "
        "RepeatMasker-style predicted-class headers. Defaults to "
        "'classified.fasta'."
    ),
)
@click.option(
    "--threshold", "-t",
    default=0.7,
    show_default=True,
    type=float,
    help="Confidence threshold for classification.",
)
@click.option(
    "--threshold-score",
    type=click.Choice(["sf_conf", "hier_conf"], case_sensitive=False),
    default="sf_conf",
    show_default=True,
    help="Confidence score used for --threshold classification/unknown calls.",
)
@click.option(
    "--hf-repo-id",
    default=DEFAULT_HF_REPO_ID,
    show_default=True,
    hidden=True,
    help="Hugging Face Hub repository used when --checkpoint is omitted.",
)
@click.option(
    "--hf-filename",
    default=DEFAULT_HF_CHECKPOINT_FILENAME,
    show_default=True,
    hidden=True,
    help="Checkpoint filename in --hf-repo-id used when --checkpoint is omitted.",
)
@click.option(
    "--hf-revision",
    default=None,
    hidden=True,
    help="Optional Hugging Face Hub revision, branch, tag, or commit.",
)
@click.option(
    "--hf-cache-dir",
    default=None,
    hidden=True,
    help="Optional Hugging Face Hub cache directory for checkpoint downloads.",
)
@click.option(
    "--temperature-scaler",
    default=None,
    hidden=True,
    help="Path to temperature_scaler.json for post-hoc calibration.",
)
@click.option(
    "--mahal-detector",
    default=None,
    hidden=True,
    help="Path to mahal_ood.npz for Mahalanobis OOD scoring.",
)
@click.option(
    "--mahal-threshold",
    default=None,
    type=float,
    hidden=True,
    help=(
        "Mahalanobis distance threshold.  When set (and --mahal-detector is "
        "provided), sequences with mahal_score > threshold are additionally "
        "flagged as unknown (is_unknown=True) in the output CSV."
    ),
)
@click.option(
    "--ood-method",
    type=click.Choice(
        ["none", "hier_conf", "energy", "mahalanobis", "mahal", "combined"],
        case_sensitive=False,
    ),
    default="none",
    hidden=True,
    show_default=True,
    help=(
        "OOD method used for the 'classified' flag. none/hier_conf use "
        "confidence without loading Mahalanobis; mahal is an alias for mahalanobis."
    ),
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Prediction batch size per GPU process. If omitted, GLARE chooses a "
        "device-aware default and retries CUDA OOMs with smaller local batches."
    ),
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help=(
        "Prediction device: auto, cpu, cuda, cuda:all, or selected "
        "logical devices such as cuda:0,2. auto remains single-device."
    ),
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "FASTA DataLoader workers per GPU process (typically 2-4); explicit "
        "values are multiplied by active GPU count."
    ),
)
@click.option(
    "--torch-threads",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "PyTorch CPU inference threads per GPU process (typically 1-2). "
        "When using --device cpu, controls PyTorch's CPU inference thread count. "
        "If omitted, scheduler/environment defaults are preserved."
    ),
)
@click.option(
    "--artifact-staging-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    hidden=True,
    help=(
        "Opt-in node-local parent directory for a unique, automatically cleaned "
        "multi-GPU staging area containing resolved worker artifacts."
    ),
)
@click.option(
    "--profile",
    is_flag=True,
    help="Emit coordinator and per-worker monotonic phase timings.",
)
def predict(
    checkpoint: str | None,
    input: str,
    output: str,
    summary_output: str | None,
    figure_output: str | None,
    stacked_figure_output: str | None,
    fasta_output: str | None,
    threshold: float,
    threshold_score: str,
    config: str | None,
    hf_repo_id: str,
    hf_filename: str,
    hf_revision: str | None,
    hf_cache_dir: str | None,
    temperature_scaler: str | None,
    mahal_detector: str | None,
    mahal_threshold: float | None,
    ood_method: str,
    batch_size: int | None,
    device: str,
    num_workers: int | None,
    torch_threads: int | None,
    artifact_staging_dir: Path | None,
    profile: bool,
) -> None:
    """Predict TE classifications for sequences in a FASTA file.

    A slim user-facing prediction CSV, summary CSV, composition plot, stacked
    composition plot, and annotated FASTA are written inside the output
    directory.
    """
    preflight_started = time.monotonic()
    device_plan = _resolve_predict_device_plan(device)
    multi_gpu_requested = len(device_plan) > 1
    if torch_threads is not None and not multi_gpu_requested:
        torch.set_num_threads(torch_threads)
    resolved_device = device_plan[0]

    from .predict import (
        build_prediction_summary,
        empty_prediction_frame,
        plot_prediction_summary,
        plot_stacked_prediction_summary,
        prediction_output_frame,
        sequence_record_offsets,
        write_annotated_fasta,
    )

    # Resolve the checkpoint before the config so local bundle directories can
    # provide inference_config.yaml without requiring --config. Explicit
    # --config and --checkpoint behavior remains unchanged.
    checkpoint_was_omitted = checkpoint is None
    checkpoint_path = _resolve_predict_checkpoint(
        checkpoint=checkpoint,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
    )

    config_path = _resolve_predict_config_for_checkpoint(
        config,
        checkpoint_path,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        allow_hf_bundle_config=checkpoint_was_omitted,
    )
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if checkpoint_was_omitted:
        _ensure_hf_bundle_snapshot(
            cfg,
            config_path,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            hf_cache_dir=hf_cache_dir,
        )
    cfg = _prepare_config_for_local_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
    )

    label_metadata = _load_bundle_label_metadata(cfg, config_path)
    if label_metadata is None and checkpoint_was_omitted:
        labels_path = _download_optional_hf_artifact(
            repo_id=hf_repo_id,
            filename=LABELS_FILENAME,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            display_name="bundled label metadata",
        )
        if labels_path is not None:
            from .predict import load_label_space_metadata

            label_metadata = load_label_space_metadata(labels_path)

    effective_batch_size, effective_num_workers = _resolve_predict_resources(
        cfg,
        resolved_device,
        batch_size,
        num_workers,
    )
    temperature_scaler = _resolve_optional_bundle_artifact(
        temperature_scaler,
        checkpoint_path,
        TEMPERATURE_SCALER_FILENAME,
        "temperature scaler",
        extra_dirs=[config_path.parent],
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        allow_hf_bundle_artifact=checkpoint_was_omitted,
    )
    normalized_ood_method = ood_method.lower()
    if normalized_ood_method == "mahal":
        normalized_ood_method = "mahalanobis"
    needs_mahalanobis = normalized_ood_method in {"mahalanobis", "combined"}
    if needs_mahalanobis:
        mahal_detector = _resolve_optional_bundle_artifact(
            mahal_detector,
            checkpoint_path,
            MAHAL_OOD_FILENAME,
            "Mahalanobis OOD detector",
            extra_dirs=[config_path.parent],
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            hf_cache_dir=hf_cache_dir,
            allow_hf_bundle_artifact=checkpoint_was_omitted,
        )
        if mahal_detector is None:
            raise click.ClickException(
                f"--ood-method {ood_method} requires --mahal-detector PATH or a "
                f"bundled {MAHAL_OOD_FILENAME} artifact."
            )
    else:
        mahal_detector = None
        if mahal_threshold is not None:
            click.echo(
                f"Warning: --mahal-threshold is ignored for --ood-method {ood_method}; "
                "no Mahalanobis artifact or scoring work will be performed.",
                err=True,
            )

    _label_space = label_metadata or str(cfg.get("data", {}).get("label_space", "panteon30"))

    input_path = Path(input)
    scan_started = time.monotonic()
    record_offsets = sequence_record_offsets(input_path)
    record_count = len(record_offsets)
    if profile:
        _profile_echo(
            "record scan", time.monotonic() - scan_started, f"records={record_count}"
        )
    ranges = _balanced_contiguous_ranges(record_count, len(device_plan))
    active_devices = device_plan[:len(ranges)]
    use_multi_gpu = len(active_devices) > 1

    if profile:
        _profile_echo(
            "preflight/artifact resolution", time.monotonic() - preflight_started
        )

    if use_multi_gpu:
        for resource_warning in _predict_resource_warnings(
            gpu_count=len(active_devices),
            num_workers=effective_num_workers,
            num_workers_explicit=num_workers is not None,
            torch_threads=torch_threads,
        ):
            click.echo(f"Warning: {resource_warning}.", err=True)
        max_batches_per_gpu = max(
            math.ceil((stop - start) / effective_batch_size) for start, stop in ranges
        )
        if max_batches_per_gpu < 50:
            click.echo(
                "Warning: multi-GPU startup may dominate this small workload "
                f"({record_count} records, at most {max_batches_per_gpu} batches/GPU; "
                "heuristic threshold: 50 batches/GPU).",
                err=True,
            )

    if use_multi_gpu:
        click.echo(
            "Devices: " + ", ".join(str(selected_device) for selected_device in active_devices)
        )
        click.echo(f"Active GPU processes: {len(active_devices)}")
        click.echo(f"Batch size per GPU: {effective_batch_size}")
        click.echo(
            f"DataLoader workers per GPU: {effective_num_workers} "
            f"(aggregate: {effective_num_workers * len(active_devices)})"
        )
        if torch_threads is not None:
            click.echo(
                f"PyTorch CPU threads per GPU: {torch_threads} "
                f"(aggregate: {torch_threads * len(active_devices)})"
            )
    else:
        active_device = active_devices[0] if active_devices else resolved_device
        click.echo(f"Device: {active_device}")
        click.echo(f"Batch size: {effective_batch_size}")
        click.echo(f"DataLoader workers: {effective_num_workers}")
        if torch_threads is not None:
            click.echo(f"PyTorch CPU threads: {torch_threads}")

    if record_count == 0:
        click.echo("Input contains no sequence records; skipping inference workers.")
        df = empty_prediction_frame()
    elif use_multi_gpu:
        worker_config = {
            "cfg": cfg,
            "checkpoint_path": str(checkpoint_path),
            "temperature_scaler_path": str(temperature_scaler) if temperature_scaler else None,
            "mahal_detector_path": str(mahal_detector) if mahal_detector else None,
            "input_path": str(input_path),
            "threshold": threshold,
            "batch_size": effective_batch_size,
            "batch_size_was_explicit": batch_size is not None,
            "num_workers": effective_num_workers,
            "torch_threads": torch_threads,
            "max_length": cfg["model"]["max_length"],
            "sliding_window_overlap": cfg.get("data", {}).get("sliding_window_overlap", 0.5),
            "use_hierarchical_confidence": threshold_score.lower() == "hier_conf",
            "mahal_threshold": mahal_threshold,
            "label_space": _label_space,
            "ood_method": normalized_ood_method,
        }
        df = _launch_multi_gpu_prediction(
            active_devices,
            ranges,
            record_count,
            worker_config,
            profile=profile,
            artifact_staging_dir=artifact_staging_dir,
            record_offsets=record_offsets,
        )
    else:
        active_device = active_devices[0]
        runtime_started = time.monotonic()
        model, tokenizer, ts, md = _load_prediction_runtime(
            cfg,
            checkpoint_path,
            active_device,
            Path(temperature_scaler) if temperature_scaler else None,
            Path(mahal_detector) if mahal_detector else None,
            config_path=config_path,
        )
        if profile:
            _profile_echo("model/runtime load", time.monotonic() - runtime_started)
        inference_started = time.monotonic()
        timing_metrics: dict = {}
        try:
            df, effective_batch_size = _predict_with_oom_backoff(
                model=model,
                tokenizer=tokenizer,
                temperature_scaler=ts,
                mahal_detector=md,
                fasta_path=input_path,
                device=active_device,
                threshold=threshold,
                batch_size=effective_batch_size,
                batch_size_was_explicit=batch_size is not None,
                num_workers=effective_num_workers,
                max_length=cfg["model"]["max_length"],
                sliding_window_overlap=cfg.get("data", {}).get("sliding_window_overlap", 0.5),
                use_hierarchical_confidence=threshold_score.lower() == "hier_conf",
                mahal_threshold=mahal_threshold,
                label_space=_label_space,
                ood_method=normalized_ood_method,
                timing_metrics=timing_metrics,
            )
        except _PredictCudaOOMError as exc:
            raise click.ClickException(str(exc)) from exc
        if profile:
            _profile_echo(
                "inference completion",
                time.monotonic() - inference_started,
                f"records={timing_metrics.get('record_count', record_count)}, "
                f"batches={timing_metrics.get('batch_count', 0)}, "
                f"final_batch_size={timing_metrics.get('final_batch_size', 0)}, "
                f"first_batch={timing_metrics.get('first_batch_seconds', 0.0):.3f}s",
            )

    output_started = time.monotonic()
    output_dir = Path(output)
    if output_dir.exists() and not output_dir.is_dir():
        raise click.ClickException(f"--output must be a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / DEFAULT_PREDICTION_CSV_FILENAME
    summary_path = _resolve_prediction_output_path(
        output_dir,
        summary_output,
        "class_summary.csv",
        "--summary-output",
    )
    figure_path = _resolve_prediction_output_path(
        output_dir,
        figure_output,
        "class_composition.png",
        "--figure-output",
    )
    stacked_figure_path = _resolve_prediction_output_path(
        output_dir,
        stacked_figure_output,
        "stacked_class_composition.png",
        "--stacked-figure-output",
    )
    fasta_path = _resolve_prediction_output_path(
        output_dir,
        fasta_output,
        "classified.fasta",
        "--fasta-output",
    )

    output_df = prediction_output_frame(df)
    output_df.to_csv(output_path, index=False)
    click.echo(f"Predictions saved to {output_path} ({len(df)} sequences)")

    summary_df = build_prediction_summary(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    click.echo(f"Class summary saved to {summary_path}")

    plot_prediction_summary(summary_df, figure_path)
    click.echo(f"Class composition figure saved to {figure_path}")

    plot_stacked_prediction_summary(summary_df, stacked_figure_path)
    click.echo(f"Stacked class composition figure saved to {stacked_figure_path}")

    write_annotated_fasta(Path(input), df, fasta_path)
    click.echo(f"Annotated FASTA saved to {fasta_path}")
    if profile:
        _profile_echo("final output generation", time.monotonic() - output_started)


if __name__ == "__main__":
    main()
