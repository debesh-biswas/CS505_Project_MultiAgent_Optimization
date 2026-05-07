# Exploring Optimization Methods for Automated Multi-Agent System Generation

**Authors:** Debesh Biswas · Hilton Raj · Vishnuram Ayyavu Vijayakumar · Manak Jain

An empirical study that replaces Particle Swarm Optimization (PSO) in the [SwarmAgentic](https://arxiv.org/abs/2025) framework with three alternative metaheuristics — **Ant Colony Optimization (ACO)**, **Genetic Algorithm (GA)**, and **Surrogate-Model-Based Optimization (SMBO)** — and benchmarks them on three agentic NLP tasks using open-source LLMs.

---

## Table of Contents

- [Exploring Optimization Methods for Automated Multi-Agent System Generation](#exploring-optimization-methods-for-automated-multi-agent-system-generation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Benchmarks](#benchmarks)
  - [Setup](#setup)
    - [Quickstart](#quickstart)
    - [Requirements](#requirements)
    - [TravelPlanner (TP) Evaluation Setup](#travelplanner-tp-evaluation-setup)
    - [Environment Variables](#environment-variables)
  - [Running the Experiments](#running-the-experiments)
    - [ACO (Ant Colony Optimization)](#aco-ant-colony-optimization)
      - [ACO — TravelPlanner (TP)](#aco--travelplanner-tp)
      - [ACO — Narrative Planning (NP)](#aco--narrative-planning-np)
      - [ACO — Creative Writing (CW)](#aco--creative-writing-cw)
      - [ACO — All Datasets](#aco--all-datasets)
    - [GA (Genetic Algorithm)](#ga-genetic-algorithm)
      - [GA — TravelPlanner (TP)](#ga--travelplanner-tp)
      - [GA — Narrative Planning (NP)](#ga--narrative-planning-np)
      - [GA — Creative Writing (CW)](#ga--creative-writing-cw)
      - [GA — All Datasets](#ga--all-datasets)
    - [SMBO (Surrogate-Model-Based Optimization)](#smbo-surrogate-model-based-optimization)
      - [SMBO — TravelPlanner (TP)](#smbo--travelplanner-tp)
      - [SMBO — Narrative Planning (NP)](#smbo--narrative-planning-np)
      - [SMBO — Creative Writing (CW)](#smbo--creative-writing-cw)
      - [SMBO — All Datasets](#smbo--all-datasets)
  - [CLI Reference](#cli-reference)
    - [Common Arguments](#common-arguments)
    - [Dataset-Specific Arguments](#dataset-specific-arguments)
      - [TravelPlanner (TP)](#travelplanner-tp)
      - [Narrative Planning (NP)](#narrative-planning-np)
      - [Creative Writing (CW)](#creative-writing-cw)
    - [Algorithm-Specific Arguments](#algorithm-specific-arguments)
      - [ACO](#aco)
      - [GA](#ga)
      - [SMBO](#smbo)
  - [Output Format](#output-format)

---

## Overview

[SwarmAgentic](https://arxiv.org/abs/2025) (Zhang et al., 2025) automates the construction of LLM-based multi-agent systems via PSO, jointly optimising agent roles, capabilities, and inter-agent coordination links. This project asks: **does the choice of optimisation algorithm matter?**

We implement three drop-in replacements for PSO:

| Method | Search Paradigm | Key Mechanism |
|--------|----------------|---------------|
| **ACO** | Population / pheromone | Pheromone-guided solution construction + evaporation |
| **GA** | Evolutionary | Tournament selection → crossover → mutation (DEAP) |
| **SMBO** | Surrogate-assisted | RandomForest surrogate + UCB acquisition over candidate pool |

All three are evaluated on the same three agentic benchmarks with open-source LLMs (LLaMA 3.1 8B / LLaMA 4 Maverick).

---

## Repository Structure

```
.
├── Project_ACO/              # Ant Colony Optimization (ACO)
│   ├── aco/                  # Core implementation + datasets
│   ├── TravelPlannerDB/      # Upstream evaluation toolchain (TP constraints)
│   └── run.py                # Entry point
│
├── Project_GA/               # Genetic Algorithm (GA)
│   ├── ga/                   # Core implementation + datasets
│   ├── ga_mas/               # Alternative GA implementation
│   └── run.py
│
├── Project_SMBO/             # Surrogate-Model-Based Optimization (SMBO)
│   ├── smbo/                 # Core implementation + datasets
│   └── run.py
│
├── requirements.txt
└── README.md
```

---

## Benchmarks

| ID | Full Name | Task Type | Metric |
|----|-----------|-----------|--------|
| **TP** | TravelPlanner | Sequential planning | Delivery rate, commonsense & hard constraint pass rate, final pass rate |
| **NP** | Narrative Planning | Structured plan generation | Exact match (regex vs. ground truth) |
| **CW** | Creative Writing | Open-ended generation | LLM-judge score 1–10, averaged over 5 samples per output |

---

## Setup

### Quickstart

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) (TP only) Install the TravelPlanner evaluation toolchain
pip install -r Project_ACO/TravelPlannerDB/requirements.txt

# 3) (TP only) Download + unzip the TravelPlanner database
# See: "TravelPlanner (TP) Evaluation Setup" below.
```

Choose one backend:

- **Hosted (NVIDIA NIM):** set `NVIDIA_API_KEY` (and optionally `NVIDIA_API_BASE`) and run normally.
- **Local (vLLM):** `pip install vllm`, set `LOCAL_LLM=1`, and pass a local model name via `--model` (e.g. `Qwen/Qwen2.5-7B-Instruct`).

> Note: We initially ran everything against NVIDIA NIM (`NVIDIA_API_*`), but long multi-agent runs frequently hit rate limits. For most experiments we used the local vLLM fallback with Qwen.

### Requirements

Recommended: Python 3.10+.

Dependencies are tracked in `requirements.txt`.

```bash
pip install -r requirements.txt
```

For the TravelPlannerDB evaluation toolchain (constraint checking, etc.):

```bash
pip install -r Project_ACO/TravelPlannerDB/requirements.txt
```

### TravelPlanner (TP) Evaluation Setup

TravelPlanner evaluation needs the upstream evaluation code plus its database files.

1) **Get the TravelPlanner repo**

This repository already includes it under `Project_ACO/TravelPlannerDB/`. If you prefer to use a fresh clone of the upstream repo, clone it from:

- https://github.com/OSU-NLP-Group/TravelPlanner

2) **Install TravelPlannerDB dependencies**

```bash
pip install -r Project_ACO/TravelPlannerDB/requirements.txt
```

3) **Download the TravelPlanner database (CSV files) and place it under TravelPlannerDB**

Option A (official instructions): follow the steps in `Project_ACO/TravelPlannerDB/README.md`.

Option B (direct download): download the database zip from Google Drive and unzip it into the TravelPlanner directory:

- https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link

After unzipping, you should have a `database/` folder under:

- `Project_ACO/TravelPlannerDB/database/`

### Environment Variables

Set the following environment variables in your shell (you can also use a `.env` file if your environment loads it).

> Model selection is controlled by the CLI flag `--model` (for both hosted and local modes).

**NVIDIA NIM (hosted, default)**

```env
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
```

Example (hosted NIM):

```powershell
# PowerShell
$env:NVIDIA_API_KEY = "nvapi-your-key-here"
cd Project_ACO
python run.py --tp-only --model meta/llama-4-maverick-17b-128e-instruct
```

```bash
# bash / Linux
cd Project_ACO
NVIDIA_API_KEY=nvapi-your-key-here python run.py --tp-only --model meta/llama-4-maverick-17b-128e-instruct
```

**Local vLLM fallback (used to avoid NIM rate limits)**

Set `LOCAL_LLM=1` and pass a local HuggingFace model name via `--model` (for example, `Qwen/Qwen2.5-7B-Instruct`).

```env
LOCAL_LLM=1
```

Example (local vLLM + Qwen):

```powershell
# PowerShell
$env:LOCAL_LLM = "1"
cd Project_ACO
python run.py --tp-only --model Qwen/Qwen2.5-7B-Instruct
```

```bash
# bash / Linux
cd Project_ACO
LOCAL_LLM=1 python run.py --tp-only --model Qwen/Qwen2.5-7B-Instruct
```

> Tip: Some tracks still use a hosted judge model depending on method/task. In particular, ACO Creative Writing uses an LLM-as-judge that requires `NVIDIA_API_KEY`.

---

## Running the Experiments

All entry points follow the same pattern:

```
cd Project_<METHOD>
python run.py [dataset flags] [algorithm flags] [common flags]
```

---

### ACO (Ant Colony Optimization)

```bash
cd Project_ACO
```

#### ACO — TravelPlanner (TP)

```bash
# Default run
python run.py --tp-only

# With common options
python run.py --tp-only \
  --tp-batch 20 \
  --tp-split validation \
  --n-ants 20 \
  --n-iterations 10 \
  --num-agents 5

# Stratified sampling (equal easy/medium/hard)
python run.py --tp-only \
  --tp-stratify --tp-per-level 4 \
  --tp-eval-stratify --tp-eval-per-level 8
```

#### ACO — Narrative Planning (NP)

```bash
# Single kind (trip / meeting / calendar)
python run.py --np-only --np-kind trip

# Mixed (all three kinds)
python run.py --np-only --np-mixed --np-per-kind 5
```

#### ACO — Creative Writing (CW)

```bash
# Standard eval mode
python run.py --cw-only --cw-batch 20

# Training tasks mode (5 ToT tasks)
python run.py --cw-only --cw-train --cw-train-batch 5
```

#### ACO — All Datasets

```bash
python run.py \
  --n-ants 20 --n-iterations 10 --num-agents 5 \
  --tp-batch 20 --np-mixed --np-per-kind 3 --cw-batch 20 \
  --log-dir ./runs
```

---

### GA (Genetic Algorithm)

```bash
cd Project_GA
```

#### GA — TravelPlanner (TP)

```bash
python run.py --tp-only \
  --tp-batch 20 \
  --tp-split validation \
  --pop-size 20 \
  --generations 10 \
  --cxpb 0.5 --mutpb 0.2 \
  --num-agents 4
```

#### GA — Narrative Planning (NP)

```bash
# Single kind
python run.py --np-only --np-kind trip \
  --pop-size 20 --generations 10

# Mixed
python run.py --np-only --np-mixed --np-per-kind 5 \
  --pop-size 20 --generations 10
```

#### GA — Creative Writing (CW)

```bash
python run.py --cw-only --cw-batch 20 \
  --pop-size 20 --generations 10
```

#### GA — All Datasets

```bash
python run.py \
  --pop-size 20 --generations 10 --cxpb 0.5 --mutpb 0.2 --num-agents 4 \
  --tp-batch 20 --np-mixed --np-per-kind 3 --cw-batch 20 \
  --log-dir ./runs
```

---

### SMBO (Surrogate-Model-Based Optimization)

```bash
cd Project_SMBO
```

#### SMBO — TravelPlanner (TP)

```bash
python run.py --tp-only \
  --tp-batch 20 \
  --tp-split validation \
  --n-init 30 --iterations 20 \
  --pool-size 500 --top-k 5 --kappa 1.0 \
  --num-agents 4
```

#### SMBO — Narrative Planning (NP)

```bash
# Single kind
python run.py --np-only --np-kind trip \
  --n-init 30 --iterations 20

# Mixed
python run.py --np-only --np-mixed --np-per-kind 5 \
  --n-init 30 --iterations 20
```

#### SMBO — Creative Writing (CW)

```bash
python run.py --cw-only --cw-batch 20 \
  --n-init 30 --iterations 20
```

#### SMBO — All Datasets

```bash
python run.py \
  --n-init 30 --iterations 20 --pool-size 500 --top-k 5 --kappa 1.0 --num-agents 4 \
  --tp-batch 20 --np-mixed --np-per-kind 3 --cw-batch 20 \
  --log-dir ./runs
```

---

## CLI Reference

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `meta/llama-4-maverick-17b-128e-instruct` | LLM backend for agents |
| `--num-agents` | `4` | Number of agents in each MAS configuration |
| `--log-dir` | `./runs` | Directory to write result JSON files |
| `--tp-only` | — | Run TravelPlanner only |
| `--np-only` | — | Run Narrative Planning only |
| `--cw-only` | — | Run Creative Writing only |
| `--skip-tp` | — | Skip TravelPlanner |
| `--skip-np` | — | Skip Narrative Planning |
| `--skip-cw` | — | Skip Creative Writing |

### Dataset-Specific Arguments

#### TravelPlanner (TP)

| Argument | Default | Description |
|----------|---------|-------------|
| `--tp-batch` | `20` | Optimisation mini-batch size |
| `--tp-split` | `validation` | Dataset split: `train` / `validation` / `test` |
| `--tp-batch-seed` | `42` | RNG seed for optimisation batch |
| `--tp-eval-batch` | `50` | Evaluation batch size |
| `--tp-eval-split` | `validation` | Split for evaluation |
| `--tp-eval-seed` | `99` | RNG seed for eval batch |
| `--tp-stratify` | `False` | Sample equal easy / medium / hard queries |
| `--tp-per-level` | `3` | Samples per difficulty level (stratified) |
| `--tp-eval-stratify` | `False` | Stratify the eval batch too |
| `--tp-eval-per-level` | `5` | Eval samples per difficulty level |

#### Narrative Planning (NP)

| Argument | Default | Description |
|----------|---------|-------------|
| `--np-kind` | `trip` | Task type: `trip` / `meeting` / `calendar` |
| `--np-mixed` | `False` | Mix all three NP kinds |
| `--np-per-kind` | `1` | Samples per kind in mixed mode |
| `--np-batch` | `160` | Samples for single-kind eval |
| `--np-batch-seed` | `42` | RNG seed |
| `--np-eval-per-kind` | `2` | Eval samples per kind (mixed mode) |
| `--np-eval-seed` | `99` | RNG seed for eval batch |

#### Creative Writing (CW)

| Argument | Default | Description |
|----------|---------|-------------|
| `--cw-batch` | `20` | Eval batch size (non-training mode) |
| `--cw-batch-seed` | `42` | RNG seed |
| `--cw-train` | `False` | Use 5 ToT training tasks instead of eval tasks |
| `--cw-train-batch` | `5` | How many training tasks to use (`0` = all 5) |
| `--cw-train-seed` | `42` | RNG seed for training batch |
| `--cw-eval-batch` | `30` | Eval batch size when in training mode |
| `--cw-eval-seed` | `99` | Eval batch seed |
| `--cw-judge-n-samples` | `5` | LLM judge calls per output |
| `--cw-judge-model` | `None` | Override judge model (default: LLaMA 3.1 8B) |

### Algorithm-Specific Arguments

#### ACO

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-ants` | `20` | Number of ants per iteration |
| `--n-iterations` | `10` | Number of ACO iterations |
| `--rho` | `0.1` | Pheromone evaporation rate |

#### GA

| Argument | Default | Description |
|----------|---------|-------------|
| `--pop-size` | `20` | Population size |
| `--generations` | `10` | Number of generations |
| `--cxpb` | `0.5` | Crossover probability |
| `--mutpb` | `0.2` | Mutation probability |

#### SMBO

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-init` | `30` | Random initial evaluations before surrogate fitting |
| `--iterations` | `20` | Number of surrogate-guided iterations |
| `--pool-size` | `500` | Candidate pool size for acquisition |
| `--top-k` | `5` | Candidates evaluated per iteration |
| `--kappa` | `1.0` | UCB exploration weight (higher = more exploration) |

---

## Output Format

Each run writes a JSON file to `--log-dir` (default `./runs/`) named by timestamp and datasets:

```
runs/
└── 20240507_143022_tp_np_trip_cw_aco_run.json
```

The JSON contains per-iteration logbook entries:

```json
{
  "best_config": { "agents": [...], "links": [...] },
  "best_score": 72.4,
  "history": [
    { "iter": 0, "avg": 41.2, "min": 30.1, "max": 58.6, "best": 58.6 },
    ...
  ]
}
```

Mirror copies of the latest run are written to method-specific folders:

| Method | TP | NP | CW |
|--------|----|----|-----|
| ACO | `aco_final/` | `aco_np/` | `aco_cw/` |
| GA | `ga_final/` | `ga_np/` | `ga_cw/` |
| SMBO | `smbo_final/` | `smbo_np/` | `smbo_cw/` |
