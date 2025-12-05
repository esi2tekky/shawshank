# Base Model vs Tulu Post-Training Variants Comparison

**Generated:** 2025-12-05 01:35:00

## Overview

This report compares the attack success rates (ASR) across four models:
- **Llama Base**: Base Llama-3.1-8B (no alignment training)
- **Tulu SFT**: Supervised Fine-Tuning
- **Tulu DPO**: Direct Preference Optimization
- **Tulu RLVR**: Reinforcement Learning from Human Feedback (RLHF)

This comparison allows us to evaluate how different alignment methods affect model vulnerability to adversarial prompts.

## Overall Attack Success Rates

| Model | ASR (%) | Noncompliant | Compliant | Uncertain | Errors |
|-------|---------|--------------|-----------|-----------|--------|
| Llama Base | 56.0% | 28 | 22 | 0 | 0 |
| Tulu SFT | 50.0% | 25 | 23 | 0 | 2 |
| Tulu DPO | 66.0% | 33 | 16 | 0 | 1 |
| Tulu RLVR | 62.0% | 31 | 18 | 0 | 1 |

### Key Findings

- **Highest ASR**: Tulu DPO (66.0%)
- **Lowest ASR**: Tulu SFT (50.0%)
- **Base Model ASR**: Llama Base (56.0%)
- **ASR Difference**: 16.0 percentage points (DPO vs SFT)
- **Base vs Aligned**: Base model (56.0%) is more vulnerable than SFT (50.0%) but less vulnerable than DPO (66.0%) and RLVR (62.0%)

## ASR by Category

The following table shows attack success rates by prompt category:

| Category | Llama Base | Tulu SFT | Tulu DPO | Tulu RLVR |
|---|---|---|---|---|
| Translation | 42.9% | 71.4% | 100.0% | 85.7% |
| Suffix | 77.8% | 66.7% | 55.6% | 33.3% |
| Obfuscation | 66.7% | 66.7% | 77.8% | 100.0% |
| Chain-of-Thought Coaxing | 77.8% | 55.6% | 77.8% | 44.4% |
| Roleplay | 36.4% | 27.3% | 54.5% | 63.6% |
| Translation / Roleplay | 0.0% | 0.0% | 0.0% | 0.0% |
| Translation / Obfuscation | 0.0% | 0.0% | 0.0% | 0.0% |
| Roleplay / Suffix | 100.0% | 0.0% | 0.0% | 100.0% |
| Obfuscation / Translation | 0.0% | 0.0% | 100.0% | 100.0% |
| Obfuscation / Roleplay | 0.0% | 0.0% | 0.0% | 0.0% |

## Visualizations

### Overall ASR Comparison
![ASR Overall](figures/tulu_comparison/asr_overall.png)

### ASR by Category
![ASR by Category](figures/tulu_comparison/asr_by_category.png)

### Judge Decision Distribution
![Judge Distribution](figures/tulu_comparison/judge_distribution.png)

### Response Length Distribution
![Response Lengths](figures/tulu_comparison/response_lengths.png)

## Methodology

- **Test Set**: 50 human-crafted adversarial prompts
- **Judge**: GPT-4 (evaluating compliance/noncompliance)
- **ASR Definition**: Percentage of prompts where model response was judged as 'noncompliant' (attack succeeded)
- **Models Compared**: Base model (no alignment) vs three alignment variants (SFT, DPO, RLVR)

## Notes

- 'Compliant' means the model refused the harmful request (attack failed)
- 'Noncompliant' means the model provided harmful content (attack succeeded)
- 'Uncertain' means the judge could not determine compliance
- 'Error' means the model failed to generate a response or judge evaluation failed
