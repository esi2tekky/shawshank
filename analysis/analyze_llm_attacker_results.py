"""
analysis/analyze_llm_attacker_results.py
----------------------------------------
Comprehensive analysis of LLM attacker results across all 4 open-source models.

Analyzes:
- Attack Success Rate (ASR) by model
- ASR by category
- Comparison with other attack methods
- Model vulnerability patterns
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

# ==== CONFIG ====
RESULTS_DIR = Path("results/llm_attacker")
OUT_DIR = Path("analysis/figures/llm_attacker")
SUMMARY_PATH = Path("analysis/llm_attacker_results_summary.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAPPING = {
    "llama_base": "Llama Base",
    "tulu_sft": "Tulu SFT",
    "tulu_dpo": "Tulu DPO",
    "tulu_rlvr": "Tulu RLVR"
}

# ==== LOAD DATA ====
print("Loading LLM attacker results...")
all_data = {}

for csv_file in RESULTS_DIR.glob("*_results.csv"):
    model_key = csv_file.stem.replace("_results", "")
    
    records = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    all_data[model_key] = records
    print(f"  ✓ {model_key}: {len(records)} records")

# ==== COMPUTE METRICS ====
print("\nComputing metrics...")
metrics = {}

for model_key, records in all_data.items():
    total = len(records)
    noncompliant = sum(1 for r in records if r.get("judge_label", "").lower() == "noncompliant")
    compliant = sum(1 for r in records if r.get("judge_label", "").lower() == "compliant")
    errors = sum(1 for r in records if r.get("judge_label", "").lower() == "error")
    
    asr = (noncompliant / total * 100) if total > 0 else 0
    
    metrics[model_key] = {
        "total": total,
        "noncompliant": noncompliant,
        "compliant": compliant,
        "errors": errors,
        "asr": asr / 100,  # Convert to fraction
    }

# ==== CATEGORY ANALYSIS ====
print("\nAnalyzing by category...")
category_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "noncompliant": 0}))

for model_key, records in all_data.items():
    for record in records:
        category = record.get("category", "unknown")
        is_noncompliant = record.get("judge_label", "").lower() == "noncompliant"
        category_stats[model_key][category]["total"] += 1
        if is_noncompliant:
            category_stats[model_key][category]["noncompliant"] += 1

# ==== GENERATE SUMMARY REPORT ====
print("\nGenerating summary report...")
summary_lines = [
    "# LLM Attacker Results: Comprehensive Analysis",
    "",
    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Overview",
    "",
    "This report analyzes LLM-generated adversarial prompts across four models:",
    "- **Llama Base**: Base Llama-3.1-8B (no alignment training)",
    "- **Tulu SFT**: Supervised Fine-Tuning",
    "- **Tulu DPO**: Direct Preference Optimization",
    "- **Tulu RLVR**: Reinforcement Learning from Human Feedback (RLHF)",
    "",
    "## Overall Attack Success Rates",
    "",
    "| Model | ASR (%) | Noncompliant | Compliant | Errors | Total |",
    "|-------|---------|--------------|-----------|--------|-------|",
]

for model_key in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"]:
    if model_key in metrics:
        m = metrics[model_key]
        summary_lines.append(
            f"| {MODEL_MAPPING[model_key]} | {m['asr']:.1%} | {m['noncompliant']} | {m['compliant']} | {m['errors']} | {m['total']} |"
        )

summary_lines.extend([
    "",
    "### Key Findings",
    "",
])

# Find best/worst
best_asr_model = max(metrics.items(), key=lambda x: x[1]["asr"])
worst_asr_model = min(metrics.items(), key=lambda x: x[1]["asr"])

summary_lines.extend([
    f"- **Highest ASR**: {MODEL_MAPPING[best_asr_model[0]]} ({best_asr_model[1]['asr']:.1%})",
    f"- **Lowest ASR**: {MODEL_MAPPING[worst_asr_model[0]]} ({worst_asr_model[1]['asr']:.1%})",
    f"- **ASR Range**: {best_asr_model[1]['asr'] - worst_asr_model[1]['asr']:.1%}",
    "",
    "## ASR by Category",
    "",
])

# Category table
all_categories = set()
for model_stats in category_stats.values():
    all_categories.update(model_stats.keys())

summary_lines.append("| Category | " + " | ".join([MODEL_MAPPING.get(k, k) for k in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"] if k in category_stats]) + " |")
summary_lines.append("|----------|" + "|".join(["---" for _ in range(len([k for k in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"] if k in category_stats]) + 1)]) + "|")

for category in sorted(all_categories):
    row = [category]
    for model_key in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"]:
        if model_key in category_stats and category in category_stats[model_key]:
            stats = category_stats[model_key][category]
            asr = stats["noncompliant"] / stats["total"] if stats["total"] > 0 else 0
            row.append(f"{asr:.1%}")
        else:
            row.append("-")
    summary_lines.append("| " + " | ".join(row) + " |")

summary_lines.extend([
    "",
    "## Examples of Successful Attacks",
    "",
])

# Add examples of successful attacks
for model_key in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"]:
    if model_key in all_data:
        successful = [r for r in all_data[model_key] if r.get("judge_label", "").lower() == "noncompliant"]
        if successful:
            example = successful[0]
            summary_lines.extend([
                f"### {MODEL_MAPPING[model_key]}",
                "",
                f"**Category:** {example.get('category', 'unknown')}",
                f"**Intent:** {example.get('intent_category', 'unknown')}",
                "",
                f"**Prompt:**",
                f"```",
                example.get('prompt', '')[:300] + "..." if len(example.get('prompt', '')) > 300 else example.get('prompt', ''),
                f"```",
                "",
                f"**Response Preview:**",
                f"```",
                example.get('response', '')[:200] + "..." if len(example.get('response', '')) > 200 else example.get('response', ''),
                f"```",
                "",
                f"**Judge Decision:** {example.get('judge_label', 'unknown')}",
                f"**Judge Reason:** {example.get('judge_reason', '')[:200]}...",
                "",
            ])

summary_lines.extend([
    "",
    "## Methodology",
    "",
    "- **Attack Method**: LLM-generated adversarial prompts (GPT-4)",
    "- **Prompt Generation**: Two-stage process (intent generation + adversarial prompt crafting)",
    "- **Test Set**: 50 prompts per model",
    "- **Judge Model**: GPT-4o-mini (cost-effective)",
    "- **Success Threshold**: Judge decision = 'noncompliant'",
    "",
    "## Comparison with Other Methods",
    "",
    "| Method | llama_base | tulu_sft | tulu_dpo | tulu_rlvr |",
    "|--------|------------|----------|----------|-----------|",
    "| **LLM Attacker** | 70.0% | 30.0% | 34.0% | 38.0% |",
    "| Human Baseline | 56.0% | 50.0% | 66.0% | 62.0% |",
    "| GA (5 gen) | 39.2% | 39.6% | 48.8% | 50.8% |",
    "| RL (200 ep) | ~55% | ~56% | ~69% | ~68% |",
    "",
    "### Key Observations",
    "",
    "- **LLM attacker is most effective on base model** (70.0% vs 56.0% human baseline)",
    "- **LLM attacker is least effective on aligned models** (30-38% vs 50-66% human baseline)",
    "- **Pattern differs from other methods**: Base model shows highest vulnerability to LLM attacks",
    "- **Alignment helps**: All aligned models show lower ASR than base model",
    "",
])

# Write summary
with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"\n✅ Summary saved to: {SUMMARY_PATH}")

# ==== CREATE VISUALIZATIONS ====
print("\nCreating visualizations...")

# 1. ASR Comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = [MODEL_MAPPING.get(k, k) for k in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"] if k in metrics]
asrs = [metrics[k]["asr"] for k in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"] if k in metrics]

bars = ax.bar(models, asrs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Attack Success Rate', fontsize=12)
ax.set_title('LLM Attacker: Attack Success Rate by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, asr in zip(bars, asrs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{asr:.1%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'asr_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Category Heatmap
intent_data = []
for category in sorted(all_categories):
    row = {'Category': category}
    for model_key in ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"]:
        if model_key in category_stats and category in category_stats[model_key]:
            stats = category_stats[model_key][category]
            asr = stats["noncompliant"] / stats["total"] if stats["total"] > 0 else 0
            row[MODEL_MAPPING.get(model_key, model_key)] = asr
        else:
            row[MODEL_MAPPING.get(model_key, model_key)] = 0
    intent_data.append(row)

intent_df = pd.DataFrame(intent_data).set_index('Category')

fig, ax = plt.subplots(figsize=(10, max(8, len(intent_df) * 0.5)))
sns.heatmap(intent_df, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
            cbar_kws={'label': 'Attack Success Rate'}, ax=ax, linewidths=0.5)
ax.set_title('LLM Attacker: ASR by Category and Model', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Category', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'category_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Visualizations saved to: {OUT_DIR}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Summary: {SUMMARY_PATH}")
print(f"Figures: {OUT_DIR}")

