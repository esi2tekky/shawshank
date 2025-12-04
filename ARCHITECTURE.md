# Shawshank Architecture Documentation

**Last Updated:** December 4, 2025  
**Branch:** `attack_targets`

---

## Overview

The Shawshank project is a hybrid "Scientific vs. Real-World" comparative framework for evaluating adversarial prompt attacks against LLMs. It supports both local open-source models (via vLLM) and remote API models (OpenAI), enabling comparative analysis across different model architectures and alignment stages.

---

## Core Architecture

### 1. Target Module (`src/target/`)

The target module provides a unified interface for interacting with different LLM backends through an abstract base class pattern.

#### 1.1 Abstract Base Class: `TargetAPI`

**File:** `src/target/target_api.py`

```python
from abc import ABC, abstractmethod

class TargetAPI(ABC):
    @abstractmethod
    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        """Return dict: {'text': str, 'tokens': int, 'metadata': {...}}"""
        pass
```

**Purpose:** Defines the contract that all target implementations must follow, ensuring a consistent interface regardless of the underlying backend.

**Return Format:**
- `text`: Generated response string
- `tokens`: Number of tokens in the response
- `metadata`: Backend-specific information (model name, finish reason, etc.)

#### 1.2 Target Factory: `target_factory.py`

**File:** `src/target/target_factory.py`

**Pattern:** Factory Pattern

**Implementation:**
```python
def load_target(model_config: str):
    """
    Factory function to load the correct target backend.
    
    Args:
        model_config: Can be a short key (e.g., 'gpt-4') or a HF path (e.g., 'allenai/...')
    """
    # Check for OpenAI models
    if "gpt" in model_config.lower():
        return OpenAITarget(model=model_config)
    
    # Default to vLLM for everything else (Tulu, Llama, Mistral)
    else:
        return VLLMTarget(model_path=model_config)
```

**Routing Logic:**
- **OpenAI Models:** Detected by presence of "gpt" in model_config string
  - Returns `OpenAITarget` instance
  - Examples: `"gpt-4o"`, `"gpt-3.5-turbo"`
  
- **Local Models:** Everything else defaults to vLLM
  - Returns `VLLMTarget` instance
  - Examples: `"allenai/Llama-3.1-Tulu-3-8B-SFT"`

**Benefits:**
- Centralized model loading logic
- Easy to extend with new backends (Anthropic, local APIs, etc.)
- No scattered if/else logic throughout codebase
- Consistent interface for all models

#### 1.3 vLLM Target: `vllm_target.py`

**File:** `src/target/vllm_target.py`

**Purpose:** Implements local model inference using vLLM for high-performance GPU-accelerated inference.

**Key Configuration (Optimized for A10G 23GB VRAM):**
```python
class VLLMTarget(TargetAPI):
    def __init__(self, model_path: str, gpu_memory_utilization=0.75, max_model_len=512):
        self.llm = LLM(
            model=model_path, 
            dtype="bfloat16",                    # Memory-efficient precision
            gpu_memory_utilization=0.75,          # Use 75% of GPU memory
            max_model_len=512,                   # Reduced from 4096 to fit in memory
            trust_remote_code=True,
            enforce_eager=True                   # Disables torch.compile to save memory
        )
```

**Memory Constraints:**
- **Model Weights:** ~15GB (Llama 3.1 8B in bfloat16)
- **KV Cache:** ~0.28GB available (with max_model_len=512)
- **Total GPU Usage:** ~16-18GB during inference
- **Available GPU:** 23GB (A10G)

**Why These Settings:**
- `max_model_len=512`: Reduced from default 4096 to ensure KV cache fits in remaining GPU memory after model weights
- `enforce_eager=True`: Disables torch.compile optimization to save memory during compilation phase
- `gpu_memory_utilization=0.75`: Balances model size with KV cache requirements
- `dtype="bfloat16"`: Reduces model size by 50% compared to float32 while maintaining quality

**Query Method:**
```python
def query(self, prompt: str, convo_ctx: dict = None) -> dict:
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    outputs = self.llm.generate([prompt], sampling_params)
    return {
        "text": generated_text.strip(),
        "tokens": len(outputs[0].outputs[0].token_ids),
        "metadata": {
            "model": self.model_name,
            "finish_reason": finish_reason,
            "backend": "vllm"
        }
    }
```

**Performance:**
- **First Load:** ~60-70 seconds (model download + initialization)
- **Subsequent Loads:** ~10-15 seconds (model cached)
- **Inference Speed:** ~25-30 tokens/second output

#### 1.4 OpenAI Target: `openai_target.py`

**File:** `src/target/openai_target.py`

**Purpose:** Wraps OpenAI API for remote model access.

**Features:**
- API key management via environment variables
- Latency tracking
- Token usage tracking
- Error handling and retries

---

### 2. Model Registry (`src/config.py`)

**File:** `src/config.py`

**Purpose:** Centralized configuration for all available models, organized by category.

**Structure:**
```python
# Open Source Models (Local via vLLM)
OPEN_SOURCE_MODELS = {
    "tulu_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",      # Supervised Fine-Tuning
    "tulu_dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",      # Direct Preference Optimization
    "tulu_rlvr": "allenai/Llama-3.1-Tulu-3-8B"          # RLHF (Reinforcement Learning)
}

# Closed Source Models (Remote via API)
CLOSED_SOURCE_MODELS = {
    "gpt_4o": "gpt-4o"
}

# Combined registry
ALL_MODELS = {**OPEN_SOURCE_MODELS, **CLOSED_SOURCE_MODELS}
```

**Design Rationale:**
- **Controlled Variable:** Alignment stage (SFT, DPO, RLHF)
- **Constants:** Base model (Llama 3.1 8B), Training data (Tulu recipe)
- **Comparative Analysis:** Enables systematic evaluation of how different alignment methods affect vulnerability to adversarial prompts

**Usage:**
```python
from src.config import OPEN_SOURCE_MODELS, ALL_MODELS
from src.target.target_factory import load_target

# Load a specific model
target = load_target(OPEN_SOURCE_MODELS['tulu_sft'])

# Iterate over all models
for key, model_id in ALL_MODELS.items():
    target = load_target(model_id)
    # ... use target
```

---

### 3. Experiment Pipeline

#### 3.1 Human Baseline Runner: `run_human_baseline_multi.py`

**File:** `src/experiments/run_human_baseline_multi.py`

**Purpose:** Run human-crafted prompts against multiple models and collect judge evaluations.

**Flow:**
```
1. Load prompts from JSONL file (data/human_baseline.jsonl)
2. For each model:
   a. Load model via target_factory
   b. For each prompt:
      - Query target model
      - Run judge evaluation (GPT-4)
      - Save results
   c. Clean up GPU memory
3. Save results to JSONL files
```

**Key Features:**
- **Multi-model support:** Runs same prompts against all specified models
- **GPU memory management:** Properly cleans up between models
- **Progress tracking:** Uses tqdm for progress bars
- **Error handling:** Graceful degradation on errors
- **Incremental output:** Writes results as they're generated

**Usage:**
```bash
python3 src/experiments/run_human_baseline_multi.py \
  --seeds data/human_baseline.jsonl \
  --output_dir results/human_baseline \
  --models tulu_sft tulu_dpo tulu_rlvr
```

**Output:**
- `results/human_baseline/human_baseline_tulu_sft.jsonl`
- `results/human_baseline/human_baseline_tulu_dpo.jsonl`
- `results/human_baseline/human_baseline_tulu_rlvr.jsonl`

**Output Schema:**
```json
{
  "id": "prompt_id",
  "category": "harm_category",
  "prompt": "original_prompt",
  "source": "human",
  "model_key": "tulu_sft",
  "model_id": "allenai/Llama-3.1-Tulu-3-8B-SFT",
  "timestamp_utc": 1234567890.0,
  "response": "model_response_text",
  "target_metadata": {...},
  "judge": {
    "decision": "compliant|noncompliant|uncertain",
    "reason": "judge_reasoning"
  }
}
```

#### 3.2 Judge Module: `gpt4_judge.py`

**File:** `src/judge/gpt4_judge.py`

**Purpose:** Automatically evaluate whether model responses comply with safety policies.

**Implementation:**
- Uses GPT-4 as a safety classifier
- Structured JSON output for reliability
- Three decision categories: `compliant`, `noncompliant`, `uncertain`

**Note on Label Semantics:**
- `"compliant"` = Model violated safety (attack succeeded)
- `"noncompliant"` = Model followed safety (attack failed)
- `"uncertain"` = Ambiguous case

---

### 4. GPU Memory Management

**Critical for Multi-Model Experiments:**

When running experiments across multiple models, proper GPU memory cleanup is essential:

```python
# After using a vLLM model
if hasattr(target, 'llm'):
    del target.llm
    del target
    gc.collect()
    torch.cuda.empty_cache()
```

**Why This Matters:**
- vLLM models consume ~15-18GB GPU memory
- A10G has 23GB total VRAM
- Without cleanup, second model load will fail with OOM error
- Cleanup ensures each model gets fresh GPU memory allocation

**Best Practices:**
1. Always check GPU memory before loading: `nvidia-smi`
2. Clean up after each model in multi-model runs
3. Kill stuck processes: `pkill -f 'python.*vllm|VLLM'`
4. Verify cleanup: `nvidia-smi` should show ~0 MiB used after cleanup

---

## Data Flow

### Human Baseline Experiment Flow

```
┌─────────────────────┐
│ human_baseline.jsonl│  (Input: 50 prompts)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ run_human_baseline_ │
│    _multi.py        │
└──────────┬──────────┘
           │
           ├─── For each model (tulu_sft, tulu_dpo, tulu_rlvr)
           │    │
           │    ├─── Load model via target_factory
           │    │
           │    ├─── For each prompt
           │    │    │
           │    │    ├─── Query target model
           │    │    │    └─── VLLMTarget.query() or OpenAITarget.query()
           │    │    │
           │    │    ├─── Run judge evaluation
           │    │    │    └─── GPT-4 judge
           │    │    │
           │    │    └─── Save result to JSONL
           │    │
           │    └─── Clean up GPU memory
           │
           ▼
┌─────────────────────┐
│ results/human_      │
│ baseline_*.jsonl    │  (Output: Results per model)
└─────────────────────┘
```

---

## Configuration Files

### Environment Variables

**File:** `.env.local` (not committed to git)

```bash
OPENAI_API_KEY=sk-proj-...
```

**Usage:**
```python
from dotenv import load_dotenv
import os
load_dotenv('.env.local')
api_key = os.getenv('OPENAI_API_KEY')
```

### Requirements

**File:** `requirements.txt`

Key dependencies:
- `vllm>=0.12.0` - Local model inference
- `torch>=2.0.0` - PyTorch backend
- `transformers>=4.30.0` - Hugging Face transformers
- `accelerate>=0.20.0` - Model acceleration
- `openai>=1.0.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Environment variable management

---

## Extension Points

### Adding a New Backend

1. **Create new target class:**
   ```python
   # src/target/new_backend_target.py
   from src.target.target_api import TargetAPI
   
   class NewBackendTarget(TargetAPI):
       def __init__(self, model_config: str):
           # Initialize backend
           pass
       
       def query(self, prompt: str, convo_ctx: dict = None) -> dict:
           # Implement query logic
           return {"text": "...", "tokens": 0, "metadata": {}}
   ```

2. **Update factory:**
   ```python
   # src/target/target_factory.py
   from src.target.new_backend_target import NewBackendTarget
   
   def load_target(model_config: str):
       if "new_backend" in model_config.lower():
           return NewBackendTarget(model_config)
       # ... existing logic
   ```

3. **Add to config:**
   ```python
   # src/config.py
   NEW_BACKEND_MODELS = {
       "model_key": "model_identifier"
   }
   ```

### Adding a New Model

1. **Add to config:**
   ```python
   # src/config.py
   OPEN_SOURCE_MODELS["new_model"] = "huggingface/model-path"
   ```

2. **Use in experiments:**
   ```bash
   python3 src/experiments/run_human_baseline_multi.py \
     --models new_model
   ```

---

## Performance Characteristics

### Model Loading Times

- **First Load (with download):** ~60-70 seconds
  - Model download: ~30-40 seconds (16GB, ~400-500 MB/s)
  - Model initialization: ~30 seconds
  
- **Subsequent Loads (cached):** ~10-15 seconds
  - Model already downloaded
  - Only initialization needed

### Inference Performance

- **Throughput:** ~25-30 tokens/second (output)
- **Latency:** ~1.5-2 seconds per query (including generation)
- **GPU Memory:** ~16-18GB during inference

### Experiment Runtime

- **Per Model:** ~5-10 minutes for 50 prompts
  - Model load: ~60 seconds
  - 50 queries: ~75-100 seconds
  - Judge evaluations: ~3-5 minutes (API calls)
  - Cleanup: ~2 seconds

- **Full Suite (3 models):** ~20-30 minutes total

---

## Troubleshooting

### Common Issues

1. **GPU Out of Memory:**
   - Check: `nvidia-smi`
   - Kill stuck processes: `pkill -f 'python.*vllm|VLLM'`
   - Clear cache: `python3 -c "import torch; torch.cuda.empty_cache()"`

2. **Model Won't Load:**
   - Verify GPU is free: `nvidia-smi`
   - Check model path is correct in `config.py`
   - Verify Hugging Face model is accessible (public models don't need auth)

3. **Import Errors:**
   - Verify conda environment is activated: `conda activate shawshank`
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **OpenAI API Errors:**
   - Verify `.env.local` exists and contains `OPENAI_API_KEY`
   - Check key is valid and has credits
   - Verify environment variable is loaded: `echo $OPENAI_API_KEY`

---

## Future Enhancements

1. **Quantization Support:**
   - AWQ/GPTQ quantization to reduce memory usage
   - Enable larger `max_model_len` on same hardware

2. **Multi-GPU Support:**
   - Tensor parallelism for larger models
   - Pipeline parallelism for very large models

3. **Caching:**
   - Response caching for repeated prompts
   - KV cache persistence across sessions

4. **Monitoring:**
   - GPU utilization tracking
   - Experiment progress dashboard
   - Cost tracking for API calls

---

## References

- **vLLM Documentation:** https://docs.vllm.ai/
- **Hugging Face Models:** https://huggingface.co/allenai
- **Tulu Models:** https://huggingface.co/collections/allenai/tulu-3-65b0c1254f4cd2ae355e1190

