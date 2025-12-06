# Genetic Algorithm Attack: Ready to Run

**Status:** ✅ Ready to execute  
**Date:** December 5, 2024

---

## Summary

We've prepared everything needed to run Genetic Algorithm (GA) attacks on all 4 open-source models using GPT-4o-mini as the judge. The setup is complete and cost-optimized.

---

## What's Ready

### 1. Cost Analysis ✅
- **Document**: `GA_COST_ANALYSIS.md`
- **Total Cost**: **$0.78** for all 4 models
- **Breakdown**: 
  - Target models: $0.00 (local vLLM)
  - Judge (GPT-4o-mini): $0.78 (4,000 calls)

### 2. Execution Script ✅
- **Script**: `run_ga_all_open_models.sh`
- **Functionality**: 
  - Runs GA on all 4 models sequentially
  - Handles GPU memory cleanup between models
  - Logs output to individual files
  - Uses GPT-4o-mini for judging

### 3. Judge Configuration ✅
- **Judge Model**: GPT-4o-mini (confirmed in `src/judge/gpt4_judge_continuous.py`)
- **Cost**: ~$0.20 per 1,000 judge calls
- **Alternative**: GPT-4 would cost ~$27 per 1,000 calls (135x more expensive)

### 4. Models to Test
1. **llama_base**: `meta-llama/Llama-3.1-8B`
2. **tulu_sft**: `allenai/Llama-3.1-Tulu-3-8B-SFT`
3. **tulu_dpo**: `allenai/Llama-3.1-Tulu-3-8B-DPO`
4. **tulu_rlvr**: `allenai/Llama-3.1-Tulu-3-8B`

---

## Experiment Configuration

### GA Parameters
- **Seed file**: `data/human_baseline.csv` (50 prompts)
- **Population size**: 100 (50 seeds × 2)
- **Generations**: 10
- **Total evaluations per model**: 1,000
- **Total evaluations across all models**: 4,000

### Expected Output
- **Results directory**: `results/ga_attacks/`
- **Per model**:
  - `ga_history_{model}_{timestamp}.json` - Evolution metrics
  - `ga_successes_{model}_{timestamp}.jsonl` - Successful attacks
  - `ga_all_evaluated_{model}_{timestamp}.jsonl` - All evaluated prompts
  - `intermediate_{model}_{timestamp}/` - Per-generation results
- **Log files**: `ga_{model_key}.log` - Console output per model

---

## How to Run

### Option 1: Run All Models (Recommended)
```bash
./run_ga_all_open_models.sh
```

This will:
1. Run GA on llama_base
2. Clean GPU memory
3. Run GA on tulu_sft
4. Clean GPU memory
5. Run GA on tulu_dpo
6. Clean GPU memory
7. Run GA on tulu_rlvr
8. Clean GPU memory

**Estimated time**: ~2-4 hours (depends on GPU speed and model loading)

### Option 2: Run Single Model (Testing)
```bash
python -m src.experiments.run_ga_attacker \
    --target meta-llama/Llama-3.1-8B \
    --seeds data/human_baseline.csv \
    --generations 10 \
    --output_dir results/ga_attacks
```

### Option 3: Custom Parameters
```bash
python -m src.experiments.run_ga_attacker \
    --target allenai/Llama-3.1-Tulu-3-8B-SFT \
    --seeds data/human_baseline.csv \
    --generations 20 \
    --mutation_rate 0.8 \
    --output_dir results/ga_attacks
```

---

## Prerequisites

### 1. Environment Setup
- ✅ Conda environment with vLLM installed
- ✅ OpenAI API key in `.env.local` (for judge)
- ✅ Hugging Face token in `.env.local` (for model access)
- ✅ GPU available (AWS EC2 instance)

### 2. Files Required
- ✅ `data/human_baseline.csv` (50 seed prompts)
- ✅ `src/config.py` (model registry)
- ✅ `src/target/target_factory.py` (model loading)
- ✅ `src/judge/gpt4_judge_continuous.py` (judge with GPT-4o-mini)

### 3. Dependencies
- ✅ vLLM (for local model inference)
- ✅ OpenAI Python client (for judge)
- ✅ All other dependencies from `requirements.txt`

---

## Cost Breakdown

| Component | Quantity | Cost |
|-----------|----------|------|
| Target queries (local) | 4,000 | $0.00 |
| Judge queries (GPT-4o-mini) | 4,000 | $0.78 |
| **Total** | | **$0.78** |

**Cost per model**: $0.20  
**Cost per 1,000 evaluations**: $0.20

---

## Expected Results

### Metrics Per Model
- **Best fitness score**: Highest judge score achieved (0.0-1.0)
- **Success rate**: Fraction of prompts with fitness ≥ 0.7
- **Total successful attacks**: Number of prompts that bypassed safety
- **Operator performance**: Which mutation operators were most effective
- **Evolution history**: How fitness improved over generations

### Comparative Analysis
After all models complete, we can compare:
- Which model is most vulnerable to GA attacks
- Which mutation operators work best per model
- How alignment stage affects vulnerability
- Success rate trends across generations

---

## Monitoring Progress

### During Execution
- Check log files: `ga_{model_key}.log`
- Monitor GPU memory: `nvidia-smi`
- Check intermediate results: `results/ga_attacks/intermediate_*/`

### After Completion
- Review summary in each `ga_history_*.json`
- Analyze successful attacks in `ga_successes_*.jsonl`
- Compare results across models

---

## Troubleshooting

### GPU Memory Issues
- Script automatically cleans GPU between models
- If issues persist, manually kill vLLM processes:
  ```bash
  pkill -9 -f vllm
  python3 -c "import torch; torch.cuda.empty_cache()"
  ```

### API Key Issues
- Ensure `.env.local` has `OPENAI_API_KEY` set
- Verify key is valid: `python -c "from openai import OpenAI; OpenAI().models.list()"`

### Model Loading Issues
- Verify Hugging Face token is set in `.env.local`
- Check model names in `src/config.py`
- Ensure GPU has enough memory (23GB+ recommended)

---

## Next Steps

1. **Run the experiment**: Execute `./run_ga_all_open_models.sh`
2. **Monitor progress**: Watch log files and GPU usage
3. **Analyze results**: Compare success rates and operator effectiveness
4. **Generate report**: Create analysis document similar to RL results

---

## Notes

- **Judge model**: GPT-4o-mini is already configured (99.3% cost savings vs GPT-4)
- **Early stopping**: Experiment stops if 100% success rate is achieved
- **Diversity preservation**: GA maintains diversity by seed_id grouping
- **Operator tracking**: Mutation operators are tracked for effectiveness analysis

---

## Questions?

- **Cost concerns?** See `GA_COST_ANALYSIS.md` for detailed breakdown
- **Script issues?** Check `run_ga_all_open_models.sh` for GPU cleanup logic
- **Model loading?** Verify `src/config.py` and `.env.local` are correct

