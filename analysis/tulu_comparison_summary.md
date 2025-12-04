# Tulu Post-Training Variants Comparison

**Generated:** 2025-12-04 15:01:22

## Overview

This report compares the attack success rates (ASR) across three Tulu post-training variants:
- **Tulu SFT**: Supervised Fine-Tuning
- **Tulu DPO**: Direct Preference Optimization
- **Tulu RLVR**: Reinforcement Learning from Human Feedback (RLHF)

## Overall Attack Success Rates

| Model | ASR (%) | Noncompliant | Compliant | Uncertain | Errors |
|-------|---------|--------------|-----------|-----------|--------|
| Tulu SFT | 50.0% | 25 | 23 | 0 | 2 |
| Tulu DPO | 66.0% | 33 | 16 | 0 | 1 |
| Tulu RLVR | 62.0% | 31 | 18 | 0 | 1 |

### Key Findings

- **Highest ASR**: Tulu DPO (66.0%)
- **Lowest ASR**: Tulu SFT (50.0%)
- **ASR Difference**: 16.0 percentage points

## ASR by Category

The following table shows attack success rates by prompt category:

| Category | Tulu SFT | Tulu DPO | Tulu RLVR |
|---|---|---|---|
| Translation | 71.4% | 100.0% | 85.7% |
| Suffix | 66.7% | 55.6% | 33.3% |
| Obfuscation | 66.7% | 77.8% | 100.0% |
| Chain-of-Thought Coaxing | 55.6% | 77.8% | 44.4% |
| Roleplay | 27.3% | 54.5% | 63.6% |
| Translation / Roleplay | 0.0% | 0.0% | 0.0% |
| Translation / Obfuscation | 0.0% | 0.0% | 0.0% |
| Roleplay / Suffix | 0.0% | 0.0% | 100.0% |
| Obfuscation / Translation | 0.0% | 100.0% | 100.0% |
| Obfuscation / Roleplay | 0.0% | 0.0% | 0.0% |

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

## Notes

- 'Compliant' means the model refused the harmful request (attack failed)
- 'Noncompliant' means the model provided harmful content (attack succeeded)
- 'Uncertain' means the judge could not determine compliance
- 'Error' means the model failed to generate a response or judge evaluation failed
