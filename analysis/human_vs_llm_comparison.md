# Human vs LLM Baseline Comparison

## 1. Attack Performance Comparison

### Overall Attack Success Rates

| Method | Judge ASR | Human ASR | Success/Total |
|--------|-----------|-----------|---------------|
| Human Baseline | 20.00% | 14.00% | 10/50 |
| LLM-Generated | 14.00% | 10.00% | 7/50 |
| **Improvement** | **-6.00pp** | - | - |

**Relative Improvement:** -30.0%

### ASR by Category (Overlapping)

| Category | Human ASR | LLM ASR | Improvement |
|----------|-----------|---------|-------------|
| Chain-of-Thought Coaxing | 0.0% | 17.4% | +17.4pp |
| Obfuscation | 33.3% | 9.1% | -24.2pp |
| Roleplay | 27.3% | 50.0% | +22.7pp |
| Suffix | 0.0% | 0.0% | +0.0pp |
| Translation | 42.9% | 0.0% | -42.9pp |

## 2. Judge Reliability Comparison

### Agreement with Human Annotators

| Baseline | Accuracy | Cohen's κ | Samples |
|----------|----------|-----------|----------|
| Human Baseline | 86.00% | 0.507 | 50 |
| LLM Baseline | 92.00% | 0.623 | 50 |

### Judge Performance Difference

- Accuracy difference: +6.00pp
- Kappa difference: +0.116

**Interpretation:** Judge is more reliable on LLM-generated attacks.

## Visualizations

![Overall ASR Comparison](figures/comparison/overall_asr_comparison.png)

![Judge Reliability Comparison](figures/comparison/judge_reliability_comparison.png)

![Category Comparison](figures/comparison/category_comparison.png)

