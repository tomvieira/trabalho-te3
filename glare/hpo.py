"""Optuna-based hyperparameter optimisation for GLARE.

Runs single-GPU trials (no DDP) so multiple workers can share a node's GPUs
in parallel.  Each worker picks up trials from a shared SQLite Optuna study.

Usage
-----
    # Single worker (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python -m glare.hpo \\
        --base-config glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment.yaml \\
        --n-trials 40 \\
        --study-name glare_hpo \\
        --storage glare_hpo.db \\
        --output-dir glare_hpo_trials \\
        --n-epochs 10

    # Four parallel workers (one per GPU) — see run_hpo_*.sh SLURM script.
"""

import argparse
import copy
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Optuna availability check
# ---------------------------------------------------------------------------

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as _optuna_err:
    print(
        "ERROR: optuna is not installed.  Install it with:\n"
        "    pip install optuna\n"
        f"Original error: {_optuna_err}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

def _sample_hyperparameters(trial: "optuna.Trial") -> dict:
    """Sample all HPO hyperparameters for a single trial."""
    return {
        "lora_lr": trial.suggest_float("lora_lr", 1e-6, 1e-4, log=True),
        "head_lr": trial.suggest_float("head_lr", 5e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "class_weight_scheme": trial.suggest_categorical(
            "class_weight_scheme", ["none", "sqrt_inv", "inv"]
        ),
        "max_length": trial.suggest_categorical("max_length", [512, 1024, 2048, 4096, 8192]),
        "sliding_window_overlap": trial.suggest_categorical(
            "sliding_window_overlap", [0.0, 0.25, 0.5]
        ),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 1024]),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3),
    }


# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------

def _apply_hparams_to_config(config: dict, hparams: dict, n_epochs: int) -> dict:
    """Return a deep-copied config with HPO hyperparameters applied."""
    cfg = copy.deepcopy(config)

    # Training section
    cfg["training"]["lora_lr"] = hparams["lora_lr"]
    cfg["training"]["head_lr"] = hparams["head_lr"]
    cfg["training"]["weight_decay"] = hparams["weight_decay"]
    cfg["training"]["batch_size"] = hparams["batch_size"]
    cfg["training"]["label_smoothing"] = hparams["label_smoothing"]
    cfg["training"]["class_weight_scheme"] = hparams["class_weight_scheme"]
    cfg["training"]["max_epochs"] = n_epochs
    # Disable torch.compile for HPO trials — adds overhead for short runs
    cfg["training"]["use_compile"] = True
    # Disable checkpoint saving for HPO trials — saves ~6-7 GB per trial
    cfg["training"]["save_checkpoint"] = False

    # Model section
    cfg["model"]["max_length"] = hparams["max_length"]
    cfg["model"]["hidden_dim"] = hparams["hidden_dim"]
    cfg["model"]["head_dropout"] = hparams["head_dropout"]

    # Data section
    cfg["data"]["sliding_window_overlap"] = hparams["sliding_window_overlap"]

    return cfg


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _read_best_val_sf_f1(trial_output_dir: Path) -> float:
    """Read the best val_sf_f1 from train_log.csv written by train().

    Returns ``float("-inf")`` if the log is missing or malformed.
    """
    log_path = trial_output_dir / "train_log.csv"
    if not log_path.exists():
        logger.warning("train_log.csv not found at %s", log_path)
        return float("-inf")

    best = float("-inf")
    try:
        with open(log_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Only consider train_val rows (not TEST rows)
                if row.get("phase", "") != "train_val":
                    continue
                raw = row.get("sf_f1", "")
                if raw == "" or raw is None:
                    continue
                try:
                    val = float(raw)
                    if val > best:
                        best = val
                except ValueError:
                    continue
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", log_path, exc)
        return float("-inf")

    return best


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    base_config: dict,
    output_dir: Path,
    n_epochs: int,
):
    """Return an Optuna objective function closed over the base config."""

    def objective(trial: "optuna.Trial") -> float:
        hparams = _sample_hyperparameters(trial)
        trial_cfg = _apply_hparams_to_config(base_config, hparams, n_epochs)

        trial_output_dir = output_dir / f"trial_{trial.number}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        # Save the trial config for reproducibility
        with open(trial_output_dir / "trial_config.yaml", "w") as fh:
            yaml.dump(trial_cfg, fh, default_flow_style=False)

        logger.info(
            "Trial %d starting | hparams: %s | output: %s",
            trial.number,
            hparams,
            trial_output_dir,
        )

        # Ensure DDP is NOT initialised: train() checks for "RANK" in os.environ.
        # We explicitly remove DDP-related env vars so the single-GPU path is used.
        for _var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(_var, None)

        try:
            from .train import train  # local import to avoid circular issues at module level

            train(
                config=trial_cfg,
                output_dir=trial_output_dir,
                local_rank=0,
            )
        except Exception as exc:
            logger.error("Trial %d FAILED with exception: %s", trial.number, exc, exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Trial failed with: {exc}")

        best_val_sf_f1 = _read_best_val_sf_f1(trial_output_dir)

        if best_val_sf_f1 == float("-inf"):
            logger.warning(
                "Trial %d: could not read val_sf_f1 from train_log.csv — pruning.",
                trial.number,
            )
            raise optuna.exceptions.TrialPruned("No val_sf_f1 found in train_log.csv")

        logger.info(
            "Trial %d finished | best_val_sf_f1=%.4f | hparams: %s",
            trial.number,
            best_val_sf_f1,
            hparams,
        )
        return best_val_sf_f1

    return objective


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def _save_trial_summary(study: "optuna.Study", output_dir: Path) -> None:
    """Append a CSV summary of all completed trials sorted by value."""
    summary_path = output_dir / "hpo_trial_summary.csv"
    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not trials:
        return

    trials_sorted = sorted(trials, key=lambda t: t.value or float("-inf"), reverse=True)

    fieldnames = ["trial_number", "val_sf_f1"] + sorted(trials_sorted[0].params.keys())
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials_sorted:
            row = {"trial_number": t.number, "val_sf_f1": t.value}
            row.update(t.params)
            writer.writerow(row)

    logger.info("Trial summary saved to %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLARE Optuna HPO — single-GPU trials with shared SQLite study"
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base YAML config file (e.g. glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials this worker should run (default: 20)",
    )
    parser.add_argument(
        "--study-name",
        default="glare_hpo",
        help="Optuna study name (default: glare_hpo)",
    )
    parser.add_argument(
        "--storage",
        default="glare_hpo.db",
        help="SQLite DB filename for persistent study storage (default: glare_hpo.db)",
    )
    parser.add_argument(
        "--output-dir",
        default="glare_hpo_trials",
        help="Root directory for per-trial output subdirectories (default: glare_hpo_trials)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Max epochs per HPO trial (default: 5; use fewer for faster search)",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=20,
        help="Number of random startup trials before TPE kicks in (default: 10)",
    )
    parser.add_argument(
        "--n-warmup-steps",
        type=int,
        default=5,
        help="MedianPruner warmup steps before pruning is active (default: 5)",
    )
    parser.add_argument(
        "--worker-seed",
        type=int,
        default=None,
        help="Random seed for this worker's TPESampler. Use different values per worker for diversity. Default: None (random).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load base config
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        logger.error("Base config not found: %s", base_config_path)
        sys.exit(1)

    with open(base_config_path) as fh:
        base_config = yaml.safe_load(fh)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SQLite storage URL — Optuna uses WAL mode internally for concurrent access
    storage_url = f"sqlite:///{args.storage}"
    logger.info("Optuna storage: %s", storage_url)
    logger.info("Study name: %s", args.study_name)
    logger.info("Output dir: %s", output_dir)
    logger.info("Trials per worker: %d | Epochs per trial: %d", args.n_trials, args.n_epochs)

    sampler = TPESampler(seed=args.worker_seed, n_startup_trials=20)
    pruner = MedianPruner(n_startup_trials=args.n_warmup_steps, n_warmup_steps=args.n_warmup_steps)

    max_retries = 10
    for attempt in range(max_retries):
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=storage_url,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = random.uniform(0.5, 3.0)
                logger.warning(
                    "Study creation attempt %d failed (%s), retrying in %.1fs...",
                    attempt + 1, e, sleep_time,
                )
                time.sleep(sleep_time)
            else:
                raise

    objective = make_objective(
        base_config=base_config,
        output_dir=output_dir,
        n_epochs=args.n_epochs,
    )

    def _after_trial_callback(
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
    ) -> None:
        """Save a summary CSV after every completed trial."""
        _save_trial_summary(study, output_dir)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[_after_trial_callback],
        catch=(Exception,),  # log failures but keep the worker alive
    )

    # Final summary
    logger.info("=" * 60)
    logger.info("HPO worker finished.  Completed trials: %d", len(study.trials))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        logger.info("Best trial: #%d | val_sf_f1=%.4f", best.number, best.value)
        logger.info("Best hyperparameters:")
        for k, v in best.params.items():
            logger.info("  %s = %s", k, v)
    else:
        logger.warning("No trials completed successfully.")

    _save_trial_summary(study, output_dir)


if __name__ == "__main__":
    main()
