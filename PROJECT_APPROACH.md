# Shawshank Project: Approach, Specifications, and Results

**Last Updated:** December 5, 2025  
**Branch:** `attack_targets`  
**Status:** Human Baseline Experiments Complete

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Approach](#research-approach)
3. [Architecture & Infrastructure](#architecture--infrastructure)
4. [Technical Specifications](#technical-specifications)
5. [Models Under Evaluation](#models-under-evaluation)
6. [Experimental Setup](#experimental-setup)
7. [Results & Findings](#results--findings)
8. [Infrastructure Details](#infrastructure-details)
9. [Next Steps](#next-steps)

---

## Project Overview

### Goal

Transform the repository from an OpenAI-only adversarial evaluation harness into a **hybrid "Scientific vs. Real-World" comparative framework**. This enables systematic evaluation of how different alignment methods affect model vulnerability to adversarial prompts.

### Research Questions

1. **Mechanism Lab (Open Source)**: How do different alignment stages (SFT, DPO, RLHF) affect vulnerability to adversarial prompts?
   - Controlled variable: Alignment stage
   - Constants: Base model (Llama 3.1 8B), Training data (Tulu recipe)

2. **Real-World Lab (Closed Source)**: How do industry-standard models (GPT-4o) compare to open-source alternatives?
   - Benchmark: Industry-standard closed-source models

### Key Innovation

- **Unified Interface**: Single codebase can evaluate both local open-source models (via vLLM) and remote API models (OpenAI)
- **Comparative Analysis**: Systematic comparison across alignment stages using identical test sets
- **Reproducibility**: All experiments are configuration-driven and fully reproducible

---

## Research Approach

### Phase 1: Infrastructure Upgrade

**Objective:** Abstract the model backend so the Attacker can interact seamlessly with both local open-source models (running on vLLM) and remote APIs (OpenAI).

**Key Components:**
- Implemented `VLLMTarget` class for local model inference
- Created `TargetFactory` pattern for unified model loading
- Established abstract `TargetAPI` interface for consistency

### Phase 2: Model Registry & Configuration

**Objective:** Define the specific models (Control vs. Variable) and create the configuration registry.

**Key Components:**
- Centralized model registry in `src/config.py`
- Organized by "Mechanism Lab" (open-source) and "Real-World Lab" (closed-source)
- Enables systematic iteration across models

### Phase 3: Human Baseline Experiments

**Objective:** Establish baseline attack success rates using human-crafted adversarial prompts.

**Methodology:**
- 50 human-crafted adversarial prompts across multiple categories
- GPT-4 as judge to evaluate compliance/noncompliance
- Run identical test set across all models
- Calculate Attack Success Rate (ASR) = % of prompts judged as "noncompliant"

### Phase 4: Comparative Analysis

**Objective:** Analyze results to understand how alignment affects vulnerability.

**Metrics:**
- Overall ASR by model
- ASR by prompt category
- Judge decision distribution
- Response length analysis

---

## Architecture & Infrastructure

### Core Design Pattern: Factory + Abstract Base Class

```
┌─────────────────┐
│  TargetAPI      │  (Abstract Base Class)
│  (Interface)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│OpenAI │ │ VLLM    │  (Concrete Implementations)
│Target │ │ Target  │
└───────┘ └─────────┘
         │
    ┌────▼────┐
    │ Factory │  (Centralized Loading)
    └─────────┘
```

### Component Overview

#### 1. Target Module (`src/target/`)

**`target_api.py`** - Abstract base class defining the contract:
```python
class TargetAPI(ABC):
    @abstractmethod
    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        """Return: {'text': str, 'tokens': int, 'metadata': {...}}"""
```

**`target_factory.py`** - Factory pattern for model loading:
- Routes OpenAI models to `OpenAITarget`
- Routes all other models to `VLLMTarget`
- Centralizes loading logic

**`vllm_target.py`** - Local model inference:
- Uses vLLM for GPU-accelerated inference
- Optimized for A10G GPU (23GB VRAM)
- Memory-efficient configuration

**`openai_target.py`** - Remote API access:
- Wraps OpenAI Python client
- Handles API key management
- Error handling and retries

#### 2. Configuration Module (`src/config.py`)

Centralized model registry:
- `OPEN_SOURCE_MODELS`: Local models (vLLM)
- `CLOSED_SOURCE_MODELS`: Remote models (OpenAI API)
- `ALL_MODELS`: Combined registry

#### 3. Experiments Module (`src/experiments/`)

**`run_human_baseline_multi.py`**:
- Iterates through specified models
- Loads each model via factory
- Queries with human-crafted prompts
- Evaluates responses using GPT-4 judge
- Saves results to JSONL files
- **Critical**: Explicit GPU memory cleanup between models

#### 4. Analysis Module (`analysis/`)

**`analyze_tulu_comparison.py`**:
- Reads JSONL results files
- Calculates ASR metrics
- Generates visualizations
- Creates summary reports

---

## Technical Specifications

### vLLM Configuration (Optimized for A10G 23GB VRAM)

```python
LLM(
    model=model_path,
    dtype="bfloat16",                    # Memory-efficient precision
    gpu_memory_utilization=0.75,          # Use 75% of GPU memory
    max_model_len=512,                   # Reduced from 4096 to fit in memory
    trust_remote_code=True,
    enforce_eager=True                   # Disables torch.compile to save memory
)
```

**Memory Breakdown:**
- Model Weights: ~15GB (Llama 3.1 8B in bfloat16)
- KV Cache: ~0.28GB (with max_model_len=512)
- Total GPU Usage: ~16-18GB during inference
- Available GPU: 23GB (A10G)

**Why These Settings:**
- `max_model_len=512`: Ensures KV cache fits in remaining GPU memory after model weights
- `enforce_eager=True`: Disables torch.compile optimization to save memory during compilation
- `gpu_memory_utilization=0.75`: Balances model size with KV cache requirements
- `dtype="bfloat16"`: Reduces model size by 50% compared to float32 while maintaining quality

### Inference Parameters

```python
SamplingParams(
    temperature=0.7,      # Matches GPT-4 default
    max_tokens=200        # Reasonable response length
)
```

### Performance Metrics

- **First Model Load**: ~60-70 seconds (model download + initialization)
- **Subsequent Loads**: ~10-15 seconds (model cached)
- **Inference Speed**: ~25-30 tokens/second output
- **Time per Model (50 prompts)**: ~5-10 minutes

### GPU Memory Management

**Critical Strategy:**
1. Load model → Run experiments → Delete model object
2. Call `gc.collect()` to free Python objects
3. Call `torch.cuda.empty_cache()` to free GPU memory
4. Small delay (5 seconds) between models

**Implementation:**
```python
if hasattr(target, 'llm'):
    del target.llm
    del target
    gc.collect()
    torch.cuda.empty_cache()
```

---

## Models Under Evaluation

### Laboratory 1: Mechanism Lab (Open Source / White Box)

**Controlled Variable:** Alignment Stage  
**Constants:** Base model (Llama 3.1 8B), Training Data (Tulu recipe)

| Model Key | Model ID | Alignment Method | Purpose |
|-----------|----------|------------------|---------|
| `llama_base` | `meta-llama/Llama-3.1-8B` | None (Base Model) | Control: No alignment training |
| `tulu_sft` | `allenai/Llama-3.1-Tulu-3-8B-SFT` | Supervised Fine-Tuning | Vulnerable instruction follower |
| `tulu_dpo` | `allenai/Llama-3.1-Tulu-3-8B-DPO` | Direct Preference Optimization | Robust/safe model (preference-based) |
| `tulu_rlvr` | `allenai/Llama-3.1-Tulu-3-8B` | RLHF (Reinforcement Learning) | State-of-the-art open source |

**Model Details:**
- **Base Architecture**: Llama 3.1 8B (8 billion parameters)
- **Training Data**: Tulu recipe (instruction-following dataset)
- **Size**: ~16GB per model (bfloat16)
- **Inference Backend**: vLLM (local GPU)

### Laboratory 2: Real-World Lab (Closed Source / Black Box)

**Purpose:** Benchmark attack severity against industry standards

| Model Key | Model ID | Provider | Purpose |
|-----------|----------|----------|---------|
| `gpt_4o` | `gpt-4o` | OpenAI | Industry-standard closed-source model |

**Model Details:**
- **Inference Backend**: OpenAI API (remote)
- **Access**: Requires API key
- **Latency**: Network-dependent (~1-3 seconds per query)

---

## Experimental Setup

### Test Set

**File:** `data/human_baseline.jsonl`  
**Size:** 50 human-crafted adversarial prompts  
**Format:** JSONL with fields:
- `id`: Unique identifier
- `category`: Prompt category (Translation, Suffix, Obfuscation, etc.)
- `prompt`: The adversarial prompt text
- `source`: "human"

**Categories:**
- Translation
- Suffix
- Obfuscation
- Chain-of-Thought Coaxing
- Roleplay
- Combined categories (Translation/Roleplay, etc.)

### Evaluation Method

**Judge:** GPT-4 (via `src/judge/gpt4_judge.py`)

**Judge Decisions:**
- `compliant`: Model refused the harmful request (attack failed)
- `noncompliant`: Model provided harmful content (attack succeeded)
- `uncertain`: Judge could not determine compliance
- `error`: Model failed to generate response or judge evaluation failed

**Attack Success Rate (ASR):**
```
ASR = (Number of "noncompliant" responses / Total responses) × 100%
```

### Experiment Execution

**Script:** `src/experiments/run_human_baseline_multi.py`

**Usage:**
```bash
python src/experiments/run_human_baseline_multi.py \
    --seeds data/human_baseline.jsonl \
    --output_dir results/human_baseline \
    --models llama_base tulu_sft tulu_dpo tulu_rlvr
```

**Output Format:**
- One JSONL file per model: `results/human_baseline/human_baseline_{model_key}.jsonl`
- Each record contains:
  - Original prompt metadata
  - Model response
  - Judge evaluation
  - Timestamp and model information

### Environment Requirements

**Local Development:**
- Python 3.10+ (required for vLLM)
- CUDA-capable GPU (for local models)
- OpenAI API key (for judge and GPT-4 target)

**Cloud Infrastructure (AWS EC2):**
- Instance Type: `g5.xlarge` (NVIDIA A10G, 23GB VRAM)
- Region: `us-west-2`
- AMI: Deep Learning Base AMI (Amazon Linux 2023)
- Storage: 50GB EBS volume
- Security: SSH access via key pair

**Dependencies:**
```
openai>=1.0.0
requests
python-dotenv
tqdm
numpy
scikit-learn
sentence-transformers
vllm
torch
transformers
accelerate
```

---

## Results & Findings

### Overall Attack Success Rates

| Model | ASR (%) | Noncompliant | Compliant | Uncertain | Errors |
|-------|---------|--------------|-----------|-----------|--------|
| **Llama Base** | **56.0%** | 28 | 22 | 0 | 0 |
| **Tulu SFT** | **50.0%** | 25 | 23 | 0 | 2 |
| **Tulu DPO** | **66.0%** | 33 | 16 | 0 | 1 |
| **Tulu RLVR** | **62.0%** | 31 | 18 | 0 | 1 |

**Key Findings:**

1. **Highest Vulnerability**: Tulu DPO (66.0% ASR)
   - Paradoxically, the "robust/safe" DPO model shows the highest attack success rate
   - Suggests DPO may create unintended vulnerabilities

2. **Lowest Vulnerability**: Tulu SFT (50.0% ASR)
   - Supervised fine-tuning provides the best protection against adversarial prompts
   - 16 percentage points lower than DPO

3. **Base Model Performance**: Llama Base (56.0% ASR)
   - More vulnerable than SFT (50.0%) but less vulnerable than DPO (66.0%) and RLVR (62.0%)
   - Suggests that alignment can both increase and decrease vulnerability depending on method

4. **ASR Range**: 16.0 percentage points (DPO vs SFT)
   - Significant variation across alignment methods
   - Indicates alignment method choice has substantial impact on security

### ASR by Category

| Category | Llama Base | Tulu SFT | Tulu DPO | Tulu RLVR |
|----------|------------|----------|----------|-----------|
| Translation | 42.9% | 71.4% | **100.0%** | 85.7% |
| Suffix | 77.8% | 66.7% | 55.6% | 33.3% |
| Obfuscation | 66.7% | 66.7% | 77.8% | **100.0%** |
| Chain-of-Thought Coaxing | 77.8% | 55.6% | 77.8% | 44.4% |
| Roleplay | 36.4% | 27.3% | 54.5% | 63.6% |

**Category-Level Insights:**

1. **Translation Attacks**: DPO is completely vulnerable (100% ASR)
   - All aligned models more vulnerable than base model
   - Suggests alignment may weaken translation-based defenses

2. **Obfuscation Attacks**: RLVR is completely vulnerable (100% ASR)
   - DPO also highly vulnerable (77.8%)
   - Base model and SFT more resistant (66.7%)

3. **Suffix Attacks**: Base model most vulnerable (77.8%)
   - RLVR most resistant (33.3%)
   - Shows alignment can improve resistance to certain attack types

4. **Chain-of-Thought Coaxing**: Base model and DPO most vulnerable (77.8%)
   - SFT and RLVR more resistant (55.6% and 44.4% respectively)

5. **Roleplay Attacks**: Base model most resistant (36.4%)
   - RLVR most vulnerable (63.6%)
   - Shows alignment can increase vulnerability to roleplay attacks

### Visualizations

Generated plots (saved to `analysis/figures/tulu_comparison/`):
1. **Overall ASR Comparison**: Bar chart comparing ASR across all models
2. **ASR by Category**: Horizontal bar chart showing category-level ASR
3. **Judge Decision Distribution**: Stacked bar chart showing decision breakdown
4. **Response Length Distribution**: Box plots comparing response lengths

### Summary Report

**Location:** `analysis/tulu_comparison_summary.md`

**Contents:**
- Overall ASR table
- Category-level ASR table
- Key findings
- Methodology notes
- Links to visualizations

---

## Infrastructure Details

### AWS EC2 Setup

**Instance Configuration:**
- **Type**: `g5.xlarge`
- **GPU**: NVIDIA A10G (23GB VRAM)
- **vCPU**: 4
- **RAM**: 16GB
- **Region**: `us-west-2`
- **IP Address**: 18.237.105.90 (may change if instance is stopped/started)
- **SSH Key**: `shawshank.pem` (located at `/Users/esi/Downloads/shawshank.pem`)

**Environment Setup:**
- **OS**: Amazon Linux 2023
- **Python**: 3.10 (via Miniconda)
- **Conda Environment**: `shawshank`
- **CUDA**: Pre-installed via Deep Learning AMI
- **NVIDIA Drivers**: Pre-installed

**Repository:**
- **Location**: `~/shawshank`
- **Branch**: `attack_targets`
- **Remote**: GitHub repository

**Environment Variables:**
- Stored in `.env.local` on instance
- `OPENAI_API_KEY`: For judge and GPT-4 target
- `HF_TOKEN`: For Hugging Face model access

### Cost Considerations

**Instance Cost:**
- `g5.xlarge`: ~$1.10/hour
- **Recommendation**: Stop instance when not in use
- **Storage**: EBS volume persists when instance is stopped

**API Costs:**
- OpenAI API: Pay-per-use for judge evaluations
- Hugging Face: Free for model downloads (with account)

### Access & Collaboration

**SSH Access:**
```bash
ssh -i ~/Downloads/shawshank.pem ec2-user@18.237.105.90
```

**Partner Access:**
- See `SSH_LOGIN_GUIDE.md` for detailed instructions
- Requires SSH key file (`shawshank.pem`)
- Key must have `chmod 400` permissions

**File Transfer:**
```bash
# Upload
scp -i ~/Downloads/shawshank.pem file.txt ec2-user@18.237.105.90:~/shawshank/

# Download
scp -i ~/Downloads/shawshank.pem ec2-user@18.237.105.90:~/shawshank/results/ results/
```

---

## Next Steps

### Immediate Tasks

1. **GPT-4 Baseline**: Run human baseline on GPT-4o to complete "Real-World Lab" comparison
   - Command: `python src/experiments/run_human_baseline_multi.py --models gpt_4o`

2. **Extended Analysis**: 
   - Compare open-source vs closed-source models
   - Analyze category-level differences
   - Investigate why DPO shows higher vulnerability

3. **Attack Development**: Begin implementing automated attack strategies
   - LLM-based attacker (GPT-4 as attacker)
   - Iterative refinement attacks
   - Compare attack effectiveness across models

### Future Enhancements

1. **Additional Models**:
   - Add more open-source models (Mistral, Qwen, etc.)
   - Add more closed-source models (Claude, Gemini, etc.)

2. **Advanced Attacks**:
   - Gradient-based attacks (for white-box models)
   - Transfer attacks (test if attacks transfer between models)
   - Adaptive attacks (attacks that adapt to model responses)

3. **Analysis Improvements**:
   - Statistical significance testing
   - Confidence intervals for ASR
   - Detailed response analysis (toxicity, helpfulness, etc.)

4. **Infrastructure**:
   - Multi-GPU support for larger models
   - Distributed evaluation across multiple instances
   - Automated experiment scheduling

### Research Questions to Explore

1. **Why is DPO more vulnerable?**
   - Hypothesis: DPO may optimize for helpfulness over safety
   - Investigation: Analyze DPO training data and objectives

2. **Category-specific vulnerabilities:**
   - Why are translation attacks so effective on DPO?
   - Why is RLVR vulnerable to obfuscation but resistant to suffix?

3. **Alignment trade-offs:**
   - Is there a fundamental trade-off between helpfulness and safety?
   - Can we design alignment methods that improve both?

---

## Documentation Files

- **`ARCHITECTURE.md`**: Detailed architecture documentation
- **`SETUP_VLLM.md`**: Complete AWS EC2 setup guide
- **`SSH_LOGIN_GUIDE.md`**: Partner SSH access instructions
- **`CHECKPOINT.md`**: Project status and troubleshooting
- **`analysis/tulu_comparison_summary.md`**: Results summary report

---

## Conclusion

We have successfully established a hybrid evaluation framework that enables systematic comparison of adversarial vulnerability across different alignment methods. The human baseline experiments reveal significant differences in attack success rates, with DPO showing unexpectedly high vulnerability (66.0% ASR) compared to SFT (50.0% ASR). These findings suggest that alignment method choice has substantial impact on model security, and that "robust" alignment methods may not always translate to better adversarial resistance.

The infrastructure is now in place to support more advanced attack strategies and broader model comparisons, setting the stage for deeper investigation into the mechanisms underlying adversarial vulnerability in aligned language models.

