"""DDP entry point for ``torchrun`` / ``srun --ntasks-per-node=4``.

Usage (via torchrun):
    torchrun --standalone --nproc_per_node=4 -m glare.train_ddp \\
        --config glare/configs/default.yaml \\
        --output-dir glare_output
"""

import argparse
import os
from pathlib import Path

import yaml

from .train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLARE DDP training entry-point (called by torchrun)"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="glare/configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="glare_output",
        help="Directory to write checkpoints and logs",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data.repbase_dir from config",
    )
    parser.add_argument(
        "--repbase-label-space",
        choices=["legacy45", "panteon30"],
        default=None,
        help="Repbase training label space: legacy 45 labels or PanTEon/GLARE 30 labels",
    )
    parser.add_argument(
        "--fasta-path",
        default=None,
        help="Override data.fasta_path from config (PanTEon mode)",
    )
    parser.add_argument(
        "--taxonomy-metadata-csv",
        default=None,
        help="Enable taxonomy holdout with this metadata CSV path",
    )
    parser.add_argument(
        "--taxonomy-filter-column",
        default=None,
        help="Metadata column to filter for taxonomy holdout, e.g. Order",
    )
    parser.add_argument(
        "--taxonomy-filter-value",
        default=None,
        help="Metadata value to hold out, e.g. Poales",
    )
    parser.add_argument(
        "--taxonomy-species-column",
        default=None,
        help="Metadata species column for taxonomy holdout, e.g. Species",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    if args.fasta_path is not None:
        config.setdefault("data", {})["fasta_path"] = args.fasta_path
    elif args.data_dir is not None:
        config["data"]["repbase_dir"] = args.data_dir
    if args.repbase_label_space is not None:
        config.setdefault("data", {})["repbase_label_space"] = args.repbase_label_space

    taxonomy_cli_values = {
        "metadata_csv": args.taxonomy_metadata_csv,
        "filter_column": args.taxonomy_filter_column,
        "filter_value": args.taxonomy_filter_value,
        "species_column": args.taxonomy_species_column,
    }
    if any(value is not None for value in taxonomy_cli_values.values()):
        missing = [key for key, value in taxonomy_cli_values.items() if value is None]
        if missing:
            raise SystemExit(
                "Taxonomy holdout CLI overrides require all of: "
                "--taxonomy-metadata-csv, --taxonomy-filter-column, "
                "--taxonomy-filter-value, --taxonomy-species-column. "
                f"Missing: {', '.join(missing)}"
            )
        holdout_cfg = config.setdefault("data", {}).setdefault("taxonomy_holdout", {})
        holdout_cfg.update({"enabled": True, **taxonomy_cli_values})

    # LOCAL_RANK is set by torchrun; fall back to 0 for single-GPU runs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    train(config=config, output_dir=Path(args.output_dir), local_rank=local_rank)


if __name__ == "__main__":
    main()
