# GLARE

**G**enomic **L**anguage model for **A**utomated **R**epetitive **E**lement classification

GLARE classifies transposable element (TE) sequences using the [NTv3-650M](https://huggingface.co/InstaDeepAI/NTv3_650M_pre) genomic language model, fine-tuned on harmonized PanTEon and Repbase reference data. Given a FASTA of candidate repeats, it assigns labels across **32 superfamilies in 11 orders** and writes a classified repeat library, per-sequence predictions, and summary figures.

## Table of contents

- [Introduction](#introduction)
- [Main results](#main-results)
- [Installation](#installation)
- [Running GLARE](#running-glare)
- [Output files](#output-files)
- [Supported repeat classes](#supported-repeat-classes)
- [Computational resources](#computational-resources)
- [Citation and contact](#citation-and-contact)
- [License](#license)
- [Copyright notice](#copyright-notice)

## Introduction

TE classification becomes difficult when a species has few close relatives in reference databases, when databases use different labels, or when repeat copies are fragmented or degraded. GLARE learns directly from nucleotide sequences and combines two reference sources in a common, RepeatMasker-style `ORDER/SUPERFAMILY` vocabulary.

For a candidate repeat library, GLARE provides:

- **Sequence-based classification:** assign repeat labels without supplying protein domains or structural features.
- **Orientation robustness:** the released model was trained with reverse-complement augmentation.
- **Confidence filtering:** report uncertain predictions as `UNKNOWN/UNKNOWN`, with a configurable threshold.
- **Ready-to-use outputs:** annotated FASTA headers, prediction and summary CSVs, and class-composition plots.

The model uses a fully fine-tuned NTv3 backbone, masked mean pooling, and a classification head that predicts superfamily. Order is derived from the predicted superfamily through a fixed label map. GLARE classifies supplied repeat sequences; TE discovery and boundary refinement are outside its scope.

## Main results

The following results are reported in the [associated preprint](https://doi.org/10.64898/2026.10.06.757139). F1 values range from 0 to 1 and are weighted by class support. Confidence filtering was disabled for evaluation, so these scores describe predictions before the default `0.7` threshold is applied.

### Released model: combined PanTEon and Repbase training

The final model uses the harmonized **32-superfamily, 11-order** vocabulary. It was trained on 282,046 records, with 31,339 validation records, and evaluated on the source-specific held-out test partitions below.

| Evaluation data | Test records | Superfamily F1 | Order F1 |
| --- | ---: | ---: | ---: |
| PanTEon | 23,043 | 0.965 | 0.992 |
| Repbase | 11,778 | 0.946 | 0.991 |
| Combined summary | 34,821 | 0.958 | 0.992 |

The combined row is a record-count-weighted average of the two source-level summaries, rather than F1 recomputed from pooled predictions.

### Comparison with existing TE classifiers

A separate GLARE model was trained for the PanTEon benchmark using its native **30-superfamily, 8-order** vocabulary and matching record partitions. GLARE achieved the highest weighted superfamily F1 among the ten classifiers evaluated. The Dfam sets contain species absent from PanTEon training and comprise 867 complete and 1,442 incomplete or degraded elements.

| Classifier | PanTEon test | Complete Dfam | Incomplete/degraded Dfam |
| --- | ---: | ---: | ---: |
| **GLARE** | **0.966** | **0.966** | **0.843** |
| NeuralTE | 0.922 | 0.934 | 0.757 |
| Terrier | 0.892 | 0.863 | 0.704 |
| CREATE | 0.887 | 0.766 | 0.710 |
| DeepTE | 0.887 | 0.736 | 0.666 |
| ClassifyTE | 0.874 | 0.624 | 0.468 |
| Inpactor2_Class | 0.847 | 0.745 | 0.654 |
| TEclass2 | 0.815 | 0.388 | 0.440 |
| TERL | 0.737 | 0.642 | 0.398 |
| BERTE | 0.539 | 0.030 | 0.094 |

### Generalization to unseen clades

Each method was retrained with the indicated clade excluded, using matched partitions within each experiment. Values are weighted superfamily F1 point estimates.

| Clade excluded from training | Taxonomic rank | GLARE | NeuralTE | Terrier |
| --- | --- | ---: | ---: | ---: |
| Poales | Order | **0.897** | 0.820 | 0.714 |
| Arthropoda | Phylum | **0.887** | 0.747 | 0.544 |
| Fungi | Kingdom | **0.783** | 0.706 | 0.447 |

### Additional findings

- **Degraded sequences:** GLARE retained the highest weighted order F1 across all tested synthetic degradation operators and severities. Under combined high-severity damage, order F1 was 0.494 for GLARE, 0.475 for NeuralTE, and 0.347 for Terrier; GLARE's superfamily F1 fell to 0.230. These perturbations are controlled stress tests, not calibrated models of natural TE decay.
- **Cross-database transfer:** with Poales withheld and a shared 25-superfamily vocabulary, combined-source training scored 0.946 on PanTEon and 0.939 on Repbase. PanTEon-only training scored 0.943 and 0.813, respectively, showing the benefit of adding Repbase for transfer between sources.
- **Backbone comparison:** under common training settings and a 512-nucleotide context, NTv3 reached 0.958 superfamily F1, compared with 0.928 for DNABERT-2 and 0.817 for Caduceus. This experiment informed backbone selection.

## Installation

### Pixi

Use [Pixi](https://pixi.sh) to create the reproducible environment defined in [pixi.toml](pixi.toml) and [pixi.lock](pixi.lock). This installs GLARE from the local checkout in editable mode. The provided environment targets **Linux x86-64**, **Python 3.11**, and **CUDA 12.4** with CUDA-enabled PyTorch. GPU execution requires a compatible NVIDIA driver.

Install Pixi if it is not already available:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

Then clone this repository and let Pixi create the environment:

```bash
git clone https://code.jgi.doe.gov/NHaas/glare
cd glare
pixi install
```

Check that the command-line interface is available:

```bash
pixi run glare --help
pixi run glare predict --help
```

## Running GLARE

### Quick start

From the repository directory, classify a FASTA containing candidate repeat sequences:

```bash
pixi run glare predict \
  --input input_sequences.fasta \
  --output predictions
```

Replace `input_sequences.fasta` with your repeat library. No input class labels are required. On first use, GLARE downloads the default checkpoint and available inference-bundle artifacts from [Dichopsis/GLARE](https://huggingface.co/Dichopsis/GLARE) and caches them through Hugging Face. Initial model setup requires internet access unless all required model and tokenizer artifacts are already available locally. The PanTEon and Repbase training databases are not needed for prediction.

The output directory is created automatically. The classified library is written to `predictions/classified.fasta`; see [Output files](#output-files) for the complete list. Use a new output directory for each run to preserve earlier results.

### Confidence threshold

The default threshold is `0.7`. Predictions below the threshold are labeled `UNKNOWN/UNKNOWN`. Increase the threshold to retain only more confident calls:

```bash
pixi run glare predict \
  --input input_sequences.fasta \
  --output predictions_threshold_08 \
  --threshold 0.8
```

Use `--threshold 0.0` to disable confidence filtering.

### Device and resource controls

By default, `--device auto` uses one CUDA device when available and otherwise CPU. It never enables multiple GPUs implicitly. Batch size and worker defaults are selected automatically; explicit controls are available when needed:

```bash
pixi run glare predict \
  --input input_sequences.fasta \
  --output predictions \
  --device auto \
  --batch-size 2 \
  --num-workers 2 \
  --torch-threads 1
```

Use one process per visible GPU with `--device cuda:all`:

```bash
# All visible GPUs
pixi run glare predict \
  --input input_sequences.fasta \
  --output predictions \
  --device cuda:all
```

Select specific visible devices with `--device cuda:0,2`, or request CPU inference with `--device cpu`. Explicit `--num-workers` and `--torch-threads` values apply per GPU process, so account for the total CPU allocation when using multiple GPUs. Add `--profile` to print timing information.

## Output files

| File | Contents |
| --- | --- |
| `predictions.csv` | Per-sequence `id`, `pred_order`, `pred_superfamily`, `repeatmasker_class`, `sf_conf`, and `is_unknown`. |
| `class_summary.csv` | Counts and proportions by predicted order and superfamily, with mean superfamily confidence. |
| `classified.fasta` | Input sequences with RepeatMasker-style `>ID#ORDER/SUPERFAMILY` headers. |
| `class_composition.png` | Class-composition bar plot. |
| `stacked_class_composition.png` | Stacked class-composition plot grouped by order. |

The annotated FASTA preserves input sequence bases while replacing the header classification, for example `>repeat_001#LTR/GYPSY`. Uncertain calls receive `>repeat_001#UNKNOWN/UNKNOWN`. Plot counts describe the supplied library's records, not the fraction of a genome occupied by each repeat class.

## Supported repeat classes

The released model uses the following 32 labels, grouped under 11 order-level categories. Satellites and structural RNAs are included to support repeat-library classification, although they are not TEs. `OTHER` is a trained category within an order; it is distinct from the confidence-filtered `UNKNOWN` label.

| Order-level category | Superfamily labels |
| --- | --- |
| `CRYPTON` | `CRYPTON` |
| `DIRS` | `DIRS` |
| `HELITRON` | `HELITRON` |
| `LINE` | `CR1`, `I`, `L1`, `OTHER`, `R1`, `RTE` |
| `LTR` | `BELPAO`, `COPIA`, `ERV`, `GYPSY`, `OTHER` |
| `MAVERICK` | `MAVERICK` |
| `PLE` | `PLE` |
| `SATELLITE` | `SATELLITE` |
| `SINE` | `7SL`, `OTHER`, `TRNA` |
| `STRUCTURAL_RNA` | `STRUCTURAL_RNA` |
| `TIR` | `ACADEM-1`, `CACTA`, `HAT`, `KOLOBOK`, `MERLIN`, `MULE`, `OTHER`, `P`, `PIFHARBINGER`, `PIGGYBAC`, `TC1MARINER` |

## Computational resources

A CUDA-capable GPU is recommended. Runtime and memory use depend on batch size, input lengths, and the number of overlapping windows. GLARE automatically retries CUDA out-of-memory failures with smaller local batches; set `--batch-size` explicitly when needed. CPU inference is supported by the CLI, but the supplied Pixi environment installs the CUDA-enabled stack.

For a typical RepeatModeler library, classifying 1,000 sequences takes **under 1 minute** on an NVIDIA A100 GPU and about **11 minutes** on 32 CPU cores.

## Citation and contact

If you use GLARE in your research, please cite our preprint:

> Nicolas Haas and Tomáš Brůna (2026). *GLARE: Genomic Language model for Automated Repetitive Element classification.* bioRxiv. https://doi.org/10.64898/2026.10.06.757139


For scientific or software questions, contact [Nicolas Haas](mailto:nhaas@lbl.gov) or [Tomáš Brůna](mailto:tbruna@lbl.gov).

## License

The GLARE source code is distributed under the terms in [LICENSE](LICENSE). Model weights have separate licensing terms, described below.

## Copyright notice

Genomic Language model for Automated Repetitive Element classification (GLARE) Copyright (c) 2026, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

### Third-party model weights

GLARE downloads fine-tuned NTv3 model weights from Hugging Face
(https://huggingface.co/Dichopsis/GLARE). These weights
are derived from [NTv3](https://github.com/instadeepai/nucleotide-transformer) and are licensed
under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).
They are not covered by the BSD-3 license of this repository.
