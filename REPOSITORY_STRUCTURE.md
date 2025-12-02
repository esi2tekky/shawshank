# Repository Structure Analysis

## Overview

The **Shawshank** repository implements a framework for evaluating different attack generation methods against LLM safety guardrails. This document provides a comprehensive analysis of the repository's structure, architecture, components, and design patterns.

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Directory Structure](#directory-structure)
3. [Component Analysis](#component-analysis)
4. [Data Flow](#data-flow)
5. [Design Patterns & Architecture Decisions](#design-patterns--architecture-decisions)
6. [Dependencies & Configuration](#dependencies--configuration)
7. [Analysis & Tools](#analysis--tools)

---

## Core Architecture

### System Overview

The Shawshank framework follows a **three-component architecture** that enables systematic evaluation of LLM safety mechanisms:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Attacker  │─────▶│    Target   │─────▶│    Judge    │
│  (Generator)│      │   (Model)   │      │ (Evaluator) │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Analysis   │
                    │   Scripts   │
                    └─────────────┘
```

### Three Main Components

#### 1. **Attacker Module** (`src/attacker/`)
Generates adversarial prompts designed to bypass LLM safety filters. Currently implements:
- **GPT-based attacker**: Two-stage generation process using GPT-4
- **Planned**: Genetic Algorithm (GA) and Reinforcement Learning (RL) attackers

#### 2. **Target Module** (`src/target/`)
Interfaces with target LLM models to test adversarial prompts. Uses an abstract base class pattern for extensibility:
- **OpenAI Target**: Wrapper for OpenAI API models (GPT-4, etc.)
- **Local Proxy**: Stub implementation for local model testing

#### 3. **Judge Module** (`src/judge/`)
Evaluates whether target model responses comply with safety policies:
- **GPT-4 Judge**: Uses GPT-4 to classify prompt-response pairs as compliant/noncompliant

### Experimental Pipeline Flow

The framework supports two main experimental pipelines:

1. **LLM Baseline Pipeline**:
   ```
   gpt_attacker.py → run_llm_attacker.py → human_annotator.py → analysis scripts
   ```

2. **Human Baseline Pipeline**:
   ```
   human prompts (JSONL) → run_human_baseline.py → human_annotator.py → analysis scripts
   ```

---

## Directory Structure

### Top-Level Organization

```
shawshank1/
├── src/              # Core source code
├── analysis/         # Analysis scripts and summaries
├── tools/            # Utility tools for data processing
├── data/             # Data storage (CSV, JSONL files)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Containerization setup
└── README.md         # Project documentation
```

### `src/` - Core Source Code

The main source code is organized into functional modules:

#### `src/attacker/` - Attack Generation
- **`gpt_attacker.py`**: Two-stage GPT-based adversarial prompt generator
  - Stage 1: Generates sensitive topics/actions within intent categories
  - Stage 2: Generates adversarial prompts using evasion strategies
  - Output: CSV with columns: `id, intent_category, intent_action, vague_action, category, prompt`
  
- **`llm_attacker.py`**: Legacy/alternative LLM attacker implementation (simpler single-stage approach)

**Key Features**:
- Supports 11 intent categories (physical harm, illegal activity, sexual content, etc.)
- Implements 5 evasion strategies (Roleplay, Translation, Suffix, Obfuscation, Chain-of-Thought)
- Uses `<INTENT>` token injection for reliable harmful intent placement
- Supports iterative refinement via feedback files

#### `src/target/` - Target Model Interface
- **`target_api.py`**: Abstract base class defining the target interface
  - Defines `query(prompt, convo_ctx)` method contract
  - Returns standardized dict: `{'text': str, 'tokens': int, 'metadata': {...}}`
  
- **`openai_target.py`**: OpenAI API implementation
  - Wraps OpenAI chat completions API
  - Tracks latency and token usage
  - Configurable model selection (GPT-4, etc.)
  
- **`local_proxy.py`**: Local testing stub
  - Implements `TargetAPI` interface
  - Placeholder for local model integration (e.g., HuggingFace models)

**Design Pattern**: Abstract base class enables easy addition of new target models (Anthropic, local models, etc.)

#### `src/judge/` - Safety Evaluation
- **`gpt4_judge.py`**: GPT-4-based safety classifier
  - Takes prompt-response pair as input
  - Returns: `{"decision": "compliant"|"noncompliant"|"uncertain", "reason": "..."}`
  - Uses structured JSON output for reliability

**Note**: The judge labels are inverted from typical safety terminology:
- `"compliant"` = model violated safety (provided harmful content)
- `"noncompliant"` = model followed safety (refused harmful request)

#### `src/experiments/` - Experiment Orchestration
- **`run_llm_attacker.py`**: LLM baseline experiment pipeline
  - Reads CSV from `gpt_attacker.py`
  - Queries target model for each prompt
  - Runs judge evaluation
  - Outputs CSV with added columns: `response, judge_label, judge_reason`
  
- **`run_human_baseline.py`**: Human baseline experiment pipeline
  - Reads JSONL of human-crafted prompts
  - Similar flow to LLM baseline
  - Outputs JSONL with judge results

#### `src/utils/` - Utility Functions
- **`storage.py`**: Simple JSON file saving utility
  - `save_result(obj, path)` function for persisting results

### `analysis/` - Analysis & Evaluation

Contains scripts for analyzing experimental results:

- **`analyze_llm_baseline.py`**: Comprehensive LLM baseline analysis
  - Computes Attack Success Rate (ASR) by evasion strategy and intent category
  - Generates visualizations (bar charts, pie charts, histograms)
  - Outputs markdown summary with figures
  
- **`analyze_human_vs_judge.py`**: Judge reliability analysis
  - Compares human annotations vs GPT-4 judge decisions
  - Computes Cohen's κ for inter-annotator agreement
  - Generates confusion matrices
  
- **`compare_human_vs_llm.py`**: Comparative analysis
  - Compares human-crafted vs LLM-generated attack performance
  - Analyzes judge accuracy on both baselines
  - Category-level breakdowns
  
- **`judge_human_comparison.py`**: Additional judge comparison utilities

**Output Files**:
- `llm_baseline_summary.md`
- `human_vs_llm_comparison.md`
- `human_vs_judge_summary.md`
- `summary.md`
- `analysis/figures/` - Generated visualization directory

### `tools/` - Utility Tools

Helper scripts for data processing and annotation:

- **`human_annotator.py`**: Interactive annotation tool
  - CLI tool for manually labeling prompt-response pairs
  - Adds `human_label` column to CSV files
  - Supports resume functionality (skips already-labeled items)
  - Keyboard shortcuts: `y`=compliant, `n`=noncompliant, `s`=skip, `q`=quit
  
- **`csv_to_jsonl.py`**: Data format conversion
  - Converts CSV files to JSONL format
  - Normalizes column names
  - Used for preparing human baseline data
  
- **`export_for_annotation.py`**: JSONL to CSV converter
  - Converts experiment results (JSONL) to CSV format
  - Extracts judge labels and reasons
  - Used for preparing data for human annotation

### `data/` - Data Storage

Stores experimental data and results:
- `v1_llm_as_attacker/feedback.txt` - Feedback files for iterative refinement
- Generated CSV/JSONL files from experiments
- Labeled datasets for analysis

**File Naming Conventions**:
- `*_baseline.csv` - Generated prompts (from attacker)
- `*_results.csv` or `*_attack.csv` - Results with target responses and judge labels
- `*_labeled.csv` - Human-annotated datasets

---

## Component Analysis

### Attacker Module

**Purpose**: Generate adversarial prompts that attempt to bypass LLM safety filters.

**Implementation Details**:

1. **Two-Stage Generation** (`gpt_attacker.py`):
   - **Stage 1**: Generate sensitive intent within predefined categories
     - Input: Intent category (e.g., "physical_harm")
     - Output: `{intent_action, vague_action}` pair
     - Uses GPT-4 with temperature=0.8 for diversity
   
   - **Stage 2**: Generate adversarial prompt using evasion strategy
     - Input: Vague action + intent action
     - Output: `{category, prompt}` with `<INTENT>` token
     - Uses GPT-4 with temperature=0.9 for creativity
     - Replaces `<INTENT>` token with actual harmful intent

2. **Intent Categories** (11 total):
   - physical_harm, illegal_activity, sexual_content, hate_speech
   - medical_advice, privacy_violation, misinformation
   - child_exploitation, acts_of_violence, creation_of_weapons, terrorism

3. **Evasion Strategies** (5 total):
   - Roleplay, Translation, Suffix, Obfuscation, Chain-of-Thought Coaxing

4. **Iterative Refinement**:
   - Supports feedback files for multi-round improvement
   - Incorporates previous round feedback into Stage 2 generation

**Planned Extensions**:
- Genetic Algorithm (GA) attacker: Optimize short suffixes/edits
- RL Editor: Policy network for iterative prompt editing

### Target Module

**Purpose**: Provide unified interface for querying different LLM models.

**Architecture**:

```python
TargetAPI (ABC)
    ├── query(prompt, convo_ctx) -> dict
    │
    ├── OpenAITarget
    │   └── Uses OpenAI API client
    │
    └── LocalProxy
        └── Stub for local models
```

**Key Design Decisions**:
- Abstract base class enables polymorphism
- Standardized return format: `{'text': str, 'tokens': int, 'metadata': dict}`
- Metadata includes latency, model info, token usage
- Easy to extend with new providers (Anthropic, local models, etc.)

### Judge Module

**Purpose**: Automatically evaluate whether model responses comply with safety policies.

**Implementation**:
- Uses GPT-4 as a safety classifier
- Structured JSON output for reliability
- Three decision categories: `compliant`, `noncompliant`, `uncertain`
- Includes reasoning for transparency

**Label Semantics** (Note: Inverted from typical usage):
- `"compliant"` = Model violated safety (attack succeeded)
- `"noncompliant"` = Model followed safety (attack failed)
- `"uncertain"` = Ambiguous case

### Experiments Module

**Purpose**: Orchestrate end-to-end experimental pipelines.

**LLM Baseline Pipeline** (`run_llm_attacker.py`):
1. Load CSV from `gpt_attacker.py`
2. For each prompt:
   - Query target model
   - Run judge evaluation
   - Record results
3. Output CSV with added columns

**Human Baseline Pipeline** (`run_human_baseline.py`):
1. Load JSONL of human-crafted prompts
2. Similar evaluation flow
3. Output JSONL with results

**Features**:
- Progress bars (tqdm)
- Error handling and graceful degradation
- Rate limiting (safety throttling)
- Incremental output (flush after each row)

---

## Data Flow

### LLM Baseline Experiment Flow

```
1. Prompt Generation (gpt_attacker.py)
   Input:  num_samples, intent_categories, evasion_strategies
   Output: data/gpt_baseline.csv
   Schema: id, intent_category, intent_action, vague_action, category, prompt

2. Target Evaluation (run_llm_attacker.py)
   Input:  data/gpt_baseline.csv
   Process: Query target model for each prompt
   Output: data/gpt_baseline_results.csv
   Schema: [original columns] + response, judge_label, judge_reason

3. Human Annotation (human_annotator.py) [Optional]
   Input:  data/gpt_baseline_results.csv
   Process: Interactive labeling
   Output: data/gpt_baseline_labeled.csv
   Schema: [previous columns] + human_label

4. Analysis (analyze_llm_baseline.py)
   Input:  data/gpt_baseline_results.csv
   Process: Compute ASR, generate visualizations
   Output: analysis/llm_baseline_summary.md, analysis/figures/
```

### Human Baseline Experiment Flow

```
1. Data Preparation
   Input:  Human-crafted prompts (CSV)
   Process: csv_to_jsonl.py converts to JSONL
   Output: data/human_baseline.jsonl

2. Target Evaluation (run_human_baseline.py)
   Input:  data/human_baseline.jsonl
   Process: Query target + judge evaluation
   Output: results/human_baseline/run.jsonl

3. Format Conversion (export_for_annotation.py)
   Input:  results/human_baseline/run.jsonl
   Process: Convert to CSV format
   Output: data/human_baseline_labeled.csv

4. Analysis (compare_human_vs_llm.py)
   Input:  Both baseline datasets
   Process: Comparative analysis
   Output: analysis/human_vs_llm_comparison.md
```

### Data Schema Evolution

**Initial Prompt CSV** (`gpt_attacker.py` output):
```csv
id, intent_category, intent_action, vague_action, category, prompt
```

**After Target Evaluation** (`run_llm_attacker.py` output):
```csv
id, intent_category, intent_action, vague_action, category, prompt, 
response, judge_label, judge_reason
```

**After Human Annotation** (`human_annotator.py` output):
```csv
[previous columns], human_label
```

---

## Design Patterns & Architecture Decisions

### 1. Abstract Base Class Pattern

**Location**: `src/target/target_api.py`

**Purpose**: Enable polymorphic target model interfaces

**Implementation**:
```python
from abc import ABC, abstractmethod

class TargetAPI(ABC):
    @abstractmethod
    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        """Return dict: {'text': str, 'tokens': int, 'metadata': {...}}"""
        pass
```

**Benefits**:
- Easy to add new target models (Anthropic, local, etc.)
- Consistent interface across implementations
- Type safety and clear contracts

### 2. Modular Component Design

**Principle**: Separation of concerns across attacker, target, and judge modules

**Benefits**:
- Independent development and testing
- Easy to swap implementations
- Clear responsibility boundaries

### 3. Configuration via Environment Variables

**Pattern**: API keys and sensitive config via environment variables

**Implementation**:
- `OPENAI_API_KEY` from environment
- `.env` file support (via `python-dotenv`)

### 4. Progressive Data Enrichment

**Pattern**: Data flows through pipeline, accumulating columns

**Flow**:
- Generation → adds prompt metadata
- Evaluation → adds response + judge labels
- Annotation → adds human labels

**Benefits**:
- Preserves original data
- Enables incremental processing
- Supports resume functionality

### 5. Error Handling & Graceful Degradation

**Pattern**: Try-except blocks with fallback values

**Examples**:
- JSON parsing failures → fallback values
- API errors → error labels in output
- Missing columns → default values

### 6. Planned vs Implemented Features

**Current Implementation**:
- ✅ Human baseline (manual prompts)
- ✅ LLM baseline (GPT-4 attacker)
- ✅ GPT-4 judge
- ✅ Analysis scripts

**Planned Features** (from README):
- ⏳ Genetic Algorithm (GA) attacker
- ⏳ Reinforcement Learning (RL) editor
- ⏳ Additional target models (Anthropic, etc.)
- ⏳ Transferability analysis

---

## Dependencies & Configuration

### Python Dependencies (`requirements.txt`)

```
openai>=1.0.0          # OpenAI API client for GPT models
requests               # HTTP library (used by OpenAI client)
python-dotenv          # Environment variable management
tqdm                   # Progress bars for CLI tools
numpy                  # Numerical computing (for analysis)
scikit-learn           # Machine learning utilities (Cohen's κ, etc.)
sentence-transformers  # Embeddings for diversity analysis
```

**Note**: Analysis scripts also use `pandas`, `matplotlib`, `seaborn` (likely installed separately or via conda)

### Docker Configuration (`Dockerfile`)

**Base Image**: `python:3.10-slim`

**Setup**:
1. Install system dependencies (git, build tools)
2. Copy requirements and install Python packages
3. Copy entire repository
4. Set `PYTHONUNBUFFERED=1` for real-time logging

**Usage**:
```bash
docker build -t shawshank:dev .
docker run -it --env-file .env -v $(pwd):/app shawshank:dev bash
```

### Environment Configuration

**Required Variables**:
- `OPENAI_API_KEY`: OpenAI API key for GPT models

**Configuration Files**:
- `.env.example`: Template for environment variables (referenced in README)
- `.env`: Actual configuration (gitignored)

### Project Structure Conventions

**Module Imports**:
- Use `python -m src.module.script` syntax to avoid import errors
- Relative imports within `src/` package

**File Naming**:
- Snake_case for Python files
- Descriptive names indicating purpose
- Suffixes: `_baseline.csv`, `_results.csv`, `_labeled.csv`

---

## Analysis & Tools

### Analysis Scripts

#### `analyze_llm_baseline.py`
**Purpose**: Comprehensive analysis of LLM-generated attack performance

**Metrics Computed**:
- Overall Attack Success Rate (ASR)
- ASR by evasion strategy
- ASR by intent category
- Judge error rate
- Response latency distribution

**Outputs**:
- Markdown summary (`llm_baseline_summary.md`)
- Visualizations in `analysis/figures/llm_baseline/`:
  - ASR by evasion strategy (bar chart)
  - ASR by intent category (bar chart)
  - Compliance breakdown (pie chart)
  - Latency histogram

#### `compare_human_vs_llm.py`
**Purpose**: Comparative analysis of human vs LLM attack methods

**Metrics Computed**:
- ASR comparison (human vs LLM)
- Judge reliability (if human labels available)
- Cohen's κ for inter-annotator agreement
- Category-level performance breakdowns

**Outputs**:
- Markdown summary (`human_vs_llm_comparison.md`)
- Visualizations in `analysis/figures/comparison/`:
  - Side-by-side ASR comparisons
  - Judge confusion matrices
  - Category-level comparisons

#### `analyze_human_vs_judge.py`
**Purpose**: Evaluate GPT-4 judge reliability against human annotations

**Metrics Computed**:
- Judge accuracy vs human labels
- Cohen's κ inter-annotator agreement
- Confusion matrix analysis
- Error pattern analysis

### Utility Tools

#### `human_annotator.py`
**Purpose**: Interactive CLI tool for manual response labeling

**Features**:
- Clear screen display of prompt-response pairs
- Keyboard shortcuts for quick labeling
- Resume functionality (skips already-labeled items)
- Progress tracking
- Incremental saving

**Usage**:
```bash
python tools/human_annotator.py --input data/gpt_baseline_results.csv
```

#### `csv_to_jsonl.py`
**Purpose**: Convert CSV files to JSONL format

**Use Case**: Preparing human-crafted prompts for the human baseline pipeline

**Features**:
- Normalizes column names (case-insensitive)
- Generates IDs if missing
- Validates required columns (prompt)

#### `export_for_annotation.py`
**Purpose**: Convert JSONL experiment results to CSV for annotation

**Use Case**: Converting `run_human_baseline.py` output to CSV format for human annotation

**Features**:
- Extracts judge labels and reasons
- Flattens nested JSON structure
- Standardizes column names

---

## Summary

The Shawshank repository demonstrates a well-structured research framework with:

1. **Clear Separation of Concerns**: Attacker, Target, and Judge modules are independently developed and tested

2. **Extensibility**: Abstract base classes and modular design enable easy addition of new attack methods, target models, and evaluation approaches

3. **Comprehensive Evaluation**: Multiple analysis scripts provide detailed insights into attack performance, judge reliability, and method comparisons

4. **Practical Tooling**: Utility scripts support the full experimental workflow from data generation to analysis

5. **Research Rigor**: Support for human annotation, inter-annotator agreement metrics, and comparative analysis ensures scientific validity

The architecture supports the project's goal of comparing different jailbreak generation approaches while maintaining code clarity, extensibility, and research reproducibility.

