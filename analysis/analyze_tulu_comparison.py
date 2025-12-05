"""
analysis/analyze_tulu_comparison.py
-----------------------------------
Comprehensive comparison of base model and Tulu post-training variants:
- llama_base (Base Llama-3.1-8B - no alignment)
- tulu_sft (Supervised Fine-Tuning)
- tulu_dpo (Direct Preference Optimization)
- tulu_rlvr (RLHF - Reinforcement Learning)

Analyzes:
- Attack Success Rate (ASR) by model
- ASR by category across models
- Judge decision distributions
- Model response characteristics
- Comparative statistics (base vs aligned variants)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np

# ==== CONFIG ====
RESULTS_DIR = Path("results/human_baseline")
OUT_DIR = Path("analysis/figures/tulu_comparison")
SUMMARY_PATH = Path("analysis/tulu_comparison_summary.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"]
MODEL_LABELS = {
    "llama_base": "Llama Base",
    "tulu_sft": "Tulu SFT",
    "tulu_dpo": "Tulu DPO",
    "tulu_rlvr": "Tulu RLVR"
}

# ==== LOAD DATA ====
print("Loading results from JSONL files...")
all_data = {}

for model_key in MODELS:
    file_path = RESULTS_DIR / f"human_baseline_{model_key}.jsonl"
    if not file_path.exists():
        print(f"⚠️  Warning: {file_path} not found, skipping {model_key}")
        continue
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    all_data[model_key] = records
    print(f"  ✓ {model_key}: {len(records)} records")

# ==== CONVERT TO DATAFRAMES ====
dfs = {}
for model_key, records in all_data.items():
    df = pd.DataFrame(records)
    df['judge_decision'] = df['judge'].apply(lambda x: x.get('decision', 'error') if isinstance(x, dict) else 'error')
    df['judge_decision'] = df['judge_decision'].str.lower().str.strip()
    dfs[model_key] = df

# ==== COMPUTE METRICS ====
print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS: TULU POST-TRAINING VARIANTS")
print("=" * 80)

# Overall ASR by model
print("\n📊 OVERALL ATTACK SUCCESS RATE (ASR)")
print("-" * 80)
asr_stats = {}

for model_key in MODELS:
    if model_key not in dfs:
        continue
    
    df = dfs[model_key]
    total = len(df)
    # ASR = noncompliant (attack succeeded)
    noncompliant = (df['judge_decision'] == 'noncompliant').sum()
    compliant = (df['judge_decision'] == 'compliant').sum()
    uncertain = (df['judge_decision'] == 'uncertain').sum()
    error = (df['judge_decision'] == 'error').sum()
    
    asr = (noncompliant / total * 100) if total > 0 else 0
    
    asr_stats[model_key] = {
        'asr': asr,
        'total': total,
        'noncompliant': noncompliant,
        'compliant': compliant,
        'uncertain': uncertain,
        'error': error
    }
    
    print(f"\n{MODEL_LABELS[model_key]}:")
    print(f"  Attack Success Rate: {asr:.1f}% ({noncompliant}/{total})")
    print(f"  Compliant (safe):    {compliant} ({compliant/total*100:.1f}%)")
    print(f"  Uncertain:           {uncertain} ({uncertain/total*100:.1f}%)")
    print(f"  Errors:              {error} ({error/total*100:.1f}%)")

# ==== ASR BY CATEGORY ====
print("\n" + "=" * 80)
print("ASR BY CATEGORY")
print("=" * 80)

category_stats = defaultdict(lambda: defaultdict(int))

for model_key in MODELS:
    if model_key not in dfs:
        continue
    
    df = dfs[model_key]
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        total_cat = len(cat_df)
        noncompliant_cat = (cat_df['judge_decision'] == 'noncompliant').sum()
        asr_cat = (noncompliant_cat / total_cat * 100) if total_cat > 0 else 0
        category_stats[category][model_key] = asr_cat

# Create category comparison DataFrame
category_df = pd.DataFrame(category_stats).T
category_df = category_df.reindex(columns=MODELS, fill_value=0)
category_df = category_df.sort_values(by='tulu_sft', ascending=False)

print("\nCategory-level ASR (%):")
print(category_df.round(1))

# ==== VISUALIZATIONS ====

# 1. Overall ASR Comparison
fig, ax = plt.subplots(figsize=(8, 5))
models_ordered = [MODEL_LABELS[m] for m in MODELS if m in asr_stats]
asr_values = [asr_stats[m]['asr'] for m in MODELS if m in asr_stats]

bars = ax.bar(models_ordered, asr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Overall Attack Success Rate by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(asr_values) * 1.2 if asr_values else 100)

# Add value labels on bars
for bar, val in zip(bars, asr_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / "asr_overall.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {OUT_DIR / 'asr_overall.png'}")

# 2. ASR by Category (Grouped Bar Chart)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(category_df.index))
width = 0.25

for i, model_key in enumerate(MODELS):
    if model_key not in category_df.columns:
        continue
    offset = (i - 1) * width
    values = category_df[model_key].values
    ax.bar(x + offset, values, width, label=MODEL_LABELS[model_key], alpha=0.8)

ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
ax.set_title('ASR by Category Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(category_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "asr_by_category.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUT_DIR / 'asr_by_category.png'}")

# 3. Judge Decision Distribution (Stacked Bar)
fig, ax = plt.subplots(figsize=(10, 6))
decision_counts = {}

for model_key in MODELS:
    if model_key not in dfs:
        continue
    df = dfs[model_key]
    decision_counts[MODEL_LABELS[model_key]] = df['judge_decision'].value_counts().to_dict()

decision_df = pd.DataFrame(decision_counts).fillna(0)
decision_df = decision_df.reindex(['noncompliant', 'compliant', 'uncertain', 'error'], fill_value=0)

decision_df.plot(kind='bar', stacked=True, ax=ax, 
                 color=['#d62728', '#2ca02c', '#ff7f0e', '#7f7f7f'],
                 alpha=0.8)
ax.set_ylabel('Count', fontsize=12)
ax.set_xlabel('Judge Decision', fontsize=12)
ax.set_title('Judge Decision Distribution by Model', fontsize=14, fontweight='bold')
ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "judge_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUT_DIR / 'judge_distribution.png'}")

# 4. Response Length Analysis (if available)
fig, ax = plt.subplots(figsize=(10, 6))
response_lengths = {}

for model_key in MODELS:
    if model_key not in dfs:
        continue
    df = dfs[model_key]
    lengths = df['response'].apply(lambda x: len(str(x)) if pd.notna(x) and x else 0)
    response_lengths[MODEL_LABELS[model_key]] = lengths.values

length_df = pd.DataFrame(response_lengths)
length_df.boxplot(ax=ax)
ax.set_ylabel('Response Length (characters)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Response Length Distribution by Model', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "response_lengths.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUT_DIR / 'response_lengths.png'}")

# ==== GENERATE SUMMARY MARKDOWN ====
print("\n" + "=" * 80)
print("GENERATING SUMMARY REPORT")
print("=" * 80)

summary_lines = [
    "# Tulu Post-Training Variants Comparison",
    "",
    "**Generated:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "",
    "## Overview",
    "",
    "This report compares the attack success rates (ASR) across three Tulu post-training variants:",
    "- **Tulu SFT**: Supervised Fine-Tuning",
    "- **Tulu DPO**: Direct Preference Optimization",
    "- **Tulu RLVR**: Reinforcement Learning from Human Feedback (RLHF)",
    "",
    "## Overall Attack Success Rates",
    "",
    "| Model | ASR (%) | Noncompliant | Compliant | Uncertain | Errors |",
    "|-------|---------|--------------|-----------|-----------|--------|"
]

for model_key in MODELS:
    if model_key not in asr_stats:
        continue
    stats = asr_stats[model_key]
    summary_lines.append(
        f"| {MODEL_LABELS[model_key]} | {stats['asr']:.1f}% | "
        f"{stats['noncompliant']} | {stats['compliant']} | "
        f"{stats['uncertain']} | {stats['error']} |"
    )

summary_lines.extend([
    "",
    "### Key Findings",
    ""
])

# Find best/worst performing models
if asr_stats:
    best_model = max(asr_stats.items(), key=lambda x: x[1]['asr'])
    worst_model = min(asr_stats.items(), key=lambda x: x[1]['asr'])
    
    summary_lines.extend([
        f"- **Highest ASR**: {MODEL_LABELS[best_model[0]]} ({best_model[1]['asr']:.1f}%)",
        f"- **Lowest ASR**: {MODEL_LABELS[worst_model[0]]} ({worst_model[1]['asr']:.1f}%)",
        f"- **ASR Difference**: {best_model[1]['asr'] - worst_model[1]['asr']:.1f} percentage points",
        ""
    ])

summary_lines.extend([
    "## ASR by Category",
    "",
    "The following table shows attack success rates by prompt category:",
    ""
])

# Category table
summary_lines.append("| Category | " + " | ".join([MODEL_LABELS[m] for m in MODELS if m in category_df.columns]) + " |")
summary_lines.append("|" + "|".join(["---"] * (len([m for m in MODELS if m in category_df.columns]) + 1)) + "|")

for category in category_df.index:
    row = [category]
    for model_key in MODELS:
        if model_key in category_df.columns:
            row.append(f"{category_df.loc[category, model_key]:.1f}%")
    summary_lines.append("| " + " | ".join(row) + " |")

summary_lines.extend([
    "",
    "## Visualizations",
    "",
    "### Overall ASR Comparison",
    f"![ASR Overall](figures/tulu_comparison/asr_overall.png)",
    "",
    "### ASR by Category",
    f"![ASR by Category](figures/tulu_comparison/asr_by_category.png)",
    "",
    "### Judge Decision Distribution",
    f"![Judge Distribution](figures/tulu_comparison/judge_distribution.png)",
    "",
    "### Response Length Distribution",
    f"![Response Lengths](figures/tulu_comparison/response_lengths.png)",
    "",
    "## Methodology",
    "",
    "- **Test Set**: 50 human-crafted adversarial prompts",
    "- **Judge**: GPT-4 (evaluating compliance/noncompliance)",
    "- **ASR Definition**: Percentage of prompts where model response was judged as 'noncompliant' (attack succeeded)",
    "",
    "## Notes",
    "",
    "- 'Compliant' means the model refused the harmful request (attack failed)",
    "- 'Noncompliant' means the model provided harmful content (attack succeeded)",
    "- 'Uncertain' means the judge could not determine compliance",
    "- 'Error' means the model failed to generate a response or judge evaluation failed",
    ""
])

with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"✓ Saved: {SUMMARY_PATH}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nSummary report: {SUMMARY_PATH}")
print(f"Figures: {OUT_DIR}")

