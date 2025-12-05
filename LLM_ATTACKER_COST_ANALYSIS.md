# LLM Attacker: Cost and Performance Analysis

**Date:** December 5, 2024  
**Experiment:** LLM-generated adversarial prompts on 4 open-source models  
**Judge Model:** GPT-4o-mini (cost-effective)

---

## Experiment Configuration

### Models Tested
1. **llama_base**: `meta-llama/Llama-3.1-8B` (local, no API cost)
2. **tulu_sft**: `allenai/Llama-3.1-Tulu-3-8B-SFT` (local, no API cost)
3. **tulu_dpo**: `allenai/Llama-3.1-Tulu-3-8B-DPO` (local, no API cost)
4. **tulu_rlvr**: `allenai/Llama-3.1-Tulu-3-8B` (local, no API cost)

### LLM Attacker Pipeline

**Stage 1: Prompt Generation** (One-time, before testing)
- Uses GPT-4 API to generate adversarial prompts
- Input: Intent categories
- Output: `data/gpt_baseline.csv` with ~50 prompts
- **Cost**: ~$0.50-1.00 (one-time, generates all prompts)

**Stage 2: Testing** (Per model)
- Loads generated prompts from CSV
- Tests each prompt on target model (local GPU via vLLM)
- Judges each response using GPT-4o-mini
- **Cost**: Judge calls only (target is free on GPU)

### Cost Components

#### 1. Prompt Generation (One-time)
- **Model**: GPT-4
- **Calls**: ~50-100 (Stage 1 + Stage 2 generation)
- **Input tokens**: ~500 per call
- **Output tokens**: ~300 per call
- **Total**: ~50,000 input + 30,000 output tokens
- **Cost**: ~$1.50-2.00 (one-time)

#### 2. Target Model Queries (4 models × ~50 prompts = 200 queries)
- **Cost**: $0.00 (all models run locally via vLLM on GPU)
- **Speed**: **Much faster on GPU** (~1-5 seconds per query vs 5-10 seconds on CPU)
- **Note**: GPU inference is ~5-10x faster than CPU

#### 3. Judge Queries (GPT-4o-mini)
- **Total judge calls**: 4 models × ~50 prompts = **~200 judge calls**

**Per Judge Call:**
- **Input tokens**: ~500 tokens (system prompt + user prompt + model response)
- **Output tokens**: ~200 tokens (JSON response with decision + reason)

**Total Tokens:**
- **Input**: 200 calls × 500 tokens = **100,000 tokens** (0.1M tokens)
- **Output**: 200 calls × 200 tokens = **40,000 tokens** (0.04M tokens)

#### 4. GPT-4o-mini Pricing (as of December 2024)
- **Input**: $0.15 per 1M tokens
- **Output**: $0.60 per 1M tokens

**Cost Calculation:**
- Input cost: 0.1M tokens × $0.15/1M = **$0.015**
- Output cost: 0.04M tokens × $0.60/1M = **$0.024**
- **Total judge cost**: **$0.039** (~$0.04)

---

## Cost Breakdown Summary

| Component | Quantity | Cost |
|-----------|----------|------|
| Prompt generation (one-time) | ~100 calls | $1.50-2.00 |
| Target model queries (local GPU) | 200 | $0.00 |
| Judge queries (GPT-4o-mini) | 200 | $0.04 |
| **Total** | | **~$1.50-2.00** |

**Cost per model**: ~$0.01 (judge only, after prompt generation)

---

## Performance: GPU vs CPU

### GPU Acceleration Benefits

**Target Model Inference (vLLM on GPU):**
- **GPU (A10G)**: ~1-5 seconds per query
- **CPU**: ~5-10 seconds per query
- **Speedup**: **5-10x faster on GPU**

**Why it's faster:**
- vLLM uses optimized CUDA kernels
- Batch processing capabilities
- KV cache optimization
- Parallel attention computation

**Total Time Estimate:**
- **With GPU**: ~50 prompts × 3 seconds = ~2.5 minutes per model
- **With CPU**: ~50 prompts × 7 seconds = ~6 minutes per model
- **Total (4 models, GPU)**: ~10 minutes
- **Total (4 models, CPU)**: ~24 minutes

**GPU saves ~14 minutes** for the full experiment.

---

## Cost Comparison: GPT-4o-mini vs GPT-4 Judge

If we used GPT-4 instead of GPT-4o-mini for judging:

**GPT-4 Pricing:**
- Input: $30.00 per 1M tokens
- Output: $60.00 per 1M tokens

**Cost Calculation:**
- Input cost: 0.1M tokens × $30.00/1M = **$3.00**
- Output cost: 0.04M tokens × $60.00/1M = **$2.40**
- **Total judge cost**: **$5.40**

**Savings by using GPT-4o-mini**: $5.40 - $0.04 = **$5.36** (99.3% cost reduction)

---

## Cost Optimization Strategies

1. **Use GPT-4o-mini for judging** ✅ (already implemented)
2. **Generate prompts once** ✅ (reuse for all models)
3. **Use local GPU for targets** ✅ (no API cost)
4. **Batch processing** (not applicable, sequential testing)

---

## Model Usage Summary

### Prompt Generation (One-time)
- **Model**: GPT-4 API
- **Calls**: ~100
- **Cost**: $1.50-2.00

### Target Models (Local vLLM on GPU)
- **4 models** × **~50 queries each** = **~200 total queries**
- **Cost**: $0.00 (local inference)
- **Speed**: ~5-10x faster than CPU
- **Compute**: GPU time on AWS EC2 (already provisioned)

### Judge Model (GPT-4o-mini API)
- **~200 judge calls**
- **Input**: 0.1M tokens
- **Output**: 0.04M tokens
- **Cost**: $0.04

---

## Time Estimates

### Prompt Generation (One-time)
- **Time**: ~5-10 minutes
- **Cost**: $1.50-2.00

### Testing on All 4 Models (GPU)
- **Per model**: ~2.5 minutes
- **Total**: ~10 minutes
- **Cost**: $0.04 (judge only)

### Total Experiment Time
- **With GPU**: ~15-20 minutes
- **With CPU**: ~30-35 minutes
- **GPU saves**: ~15 minutes

---

## Conclusion

The LLM attacker experiment on 4 open-source models is **extremely cost-effective** at **~$1.50-2.00 total** (mostly one-time prompt generation), thanks to:

1. **One-time prompt generation**: Generate once, test on all models
2. **Local GPU targets**: No API cost for target models
3. **GPT-4o-mini judge**: 99.3% cheaper than GPT-4
4. **GPU acceleration**: 5-10x faster than CPU

**Key Benefits:**
- **Fast**: ~10 minutes total testing time on GPU
- **Cheap**: ~$0.04 per model (after prompt generation)
- **Efficient**: Reuse generated prompts across all models

This makes it feasible to run multiple experimental variations and scale up if needed.

