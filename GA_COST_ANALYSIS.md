# Genetic Algorithm Attack: Cost Analysis

**Date:** December 5, 2024  
**Experiment:** GA-based adversarial prompt generation on 4 open-source models  
**Judge Model:** GPT-4o-mini (cost-effective continuous scoring)

---

## Experiment Configuration

### Models Tested
1. **llama_base**: `meta-llama/Llama-3.1-8B` (local, no API cost)
2. **tulu_sft**: `allenai/Llama-3.1-Tulu-3-8B-SFT` (local, no API cost)
3. **tulu_dpo**: `allenai/Llama-3.1-Tulu-3-8B-DPO` (local, no API cost)
4. **tulu_rlvr**: `allenai/Llama-3.1-Tulu-3-8B` (local, no API cost)

### GA Parameters
- **Seed file**: `data/human_baseline.csv` (50 prompts)
- **Population size**: 100 individuals (50 seeds × 2: 50 elites + 50 mutations)
- **Generations**: 10 (default)
- **Total evaluations per model**: 100 individuals × 10 generations = **1,000 evaluations**

### Cost Components

#### 1. Target Model Queries (4 models × 1,000 queries = 4,000 queries)
- **Cost**: $0.00 (all models run locally via vLLM)
- **Note**: GPU compute time is free (using AWS EC2 instance already provisioned)

#### 2. Judge Queries (GPT-4o-mini)
- **Total judge calls**: 4 models × 1,000 evaluations = **4,000 judge calls**

**Per Judge Call:**
- **Input tokens**: ~500 tokens (system prompt + user prompt + model response)
  - System prompt: ~400 tokens
  - User prompt: ~50 tokens
  - Model response: ~50 tokens (truncated preview)
- **Output tokens**: ~200 tokens (JSON response with score + reason)

**Total Tokens:**
- **Input**: 4,000 calls × 500 tokens = **2,000,000 tokens** (2M tokens)
- **Output**: 4,000 calls × 200 tokens = **800,000 tokens** (0.8M tokens)

#### 3. GPT-4o-mini Pricing (as of December 2024)
- **Input**: $0.15 per 1M tokens
- **Output**: $0.60 per 1M tokens

**Cost Calculation:**
- Input cost: 2.0M tokens × $0.15/1M = **$0.30**
- Output cost: 0.8M tokens × $0.60/1M = **$0.48**
- **Total judge cost**: **$0.78**

---

## Cost Breakdown Summary

| Component | Quantity | Cost |
|-----------|----------|------|
| Target model queries (local) | 4,000 | $0.00 |
| Judge queries (GPT-4o-mini) | 4,000 | $0.78 |
| **Total** | | **$0.78** |

---

## Cost Comparison: GPT-4o-mini vs GPT-4

If we used GPT-4 instead of GPT-4o-mini for judging:

**GPT-4 Pricing:**
- Input: $30.00 per 1M tokens
- Output: $60.00 per 1M tokens

**Cost Calculation:**
- Input cost: 2.0M tokens × $30.00/1M = **$60.00**
- Output cost: 0.8M tokens × $60.00/1M = **$48.00**
- **Total judge cost**: **$108.00**

**Savings by using GPT-4o-mini**: $108.00 - $0.78 = **$107.22** (99.3% cost reduction)

---

## Additional Cost Considerations

### 1. LLM Mutations (Optional)
If `use_llm_mutations=True` (default for non-GPT targets):
- Uses local models for mutation, so **no additional API cost**
- Only rule-based mutations are used (no LLM calls)

### 2. Early Stopping
- If early stopping triggers (100% success rate), actual costs will be **lower**
- Worst case: Full 10 generations = $0.78

### 3. Scaling Considerations

**If we increase generations to 20:**
- Total evaluations: 4 models × 2,000 = 8,000
- Judge cost: **$1.56**

**If we increase population size (more seeds):**
- 100 seeds → 200 population → 2,000 evaluations per model
- Total: 4 models × 2,000 = 8,000 evaluations
- Judge cost: **$1.56**

**If we test on GPT-4o as target (closed-source):**
- Target queries: 1,000 × $5.00/1M input × ~500 tokens = **$2.50** (input only, assuming short responses)
- Judge queries: 1,000 × $0.78/4,000 = **$0.20**
- **Total**: **$2.70** per GPT-4o experiment

---

## Budget Recommendation

For the full experiment suite:
- **4 open-source models** (GA): **$0.78**
- **1 closed-source model** (GPT-4o, optional): **$2.70**
- **Total recommended budget**: **~$3.50**

This is extremely cost-effective compared to using GPT-4 as judge ($108+).

---

## Cost Optimization Strategies

1. **Use GPT-4o-mini for judging** ✅ (already implemented)
2. **Early stopping** ✅ (already implemented, stops at 100% success rate)
3. **Batch processing** (not applicable for sequential GA)
4. **Cache judge results** (not applicable, each prompt is unique)
5. **Reduce generations** (if early results are sufficient)

---

## Model Usage Summary

### Target Models (Local vLLM)
- **4 models** × **1,000 queries each** = **4,000 total queries**
- **Cost**: $0.00 (local inference)
- **Compute**: GPU time on AWS EC2 (already provisioned)

### Judge Model (GPT-4o-mini API)
- **4,000 judge calls**
- **Input**: 2.0M tokens
- **Output**: 0.8M tokens
- **Cost**: $0.78

---

## Conclusion

The GA attack experiment on 4 open-source models is **extremely cost-effective** at **$0.78 total**, thanks to:
1. Using local models for targets (no API cost)
2. Using GPT-4o-mini for judging (99.3% cheaper than GPT-4)
3. Efficient population management (100 individuals, 10 generations)

This makes it feasible to run multiple experimental variations and scale up if needed.

