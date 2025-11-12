"""
analysis/compare_human_vs_llm.py
---------------------------------
Compare Human Baseline vs LLM-Generated Baseline attack performance.

Compares:
- Human baseline: data/human_baseline_labeled.csv
- LLM baseline: data/gpt_baseline_attack.csv

Generates:
- Side-by-side ASR comparison
- Category overlap analysis
- Performance improvement metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==== CONFIG ====
HUMAN_CSV = "data/human_baseline_labeled.csv"
LLM_CSV = "data/gpt_baseline_attack.csv"
OUT_DIR = Path("analysis/figures/comparison")
SUMMARY_PATH = Path("analysis/human_vs_llm_comparison.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== LOAD DATA ====
print("📂 Loading human baseline...")
human_df = pd.read_csv(HUMAN_CSV)
human_df["judge_label"] = human_df["judge_label"].fillna("error").str.lower()

print("📂 Loading LLM baseline...")
llm_df = pd.read_csv(LLM_CSV)
llm_df["judge_label"] = llm_df["judge_label"].fillna("error").str.lower()

# ==== COMPUTE OVERALL ASR ====
human_total = len(human_df)
human_success = (human_df["judge_label"] == "noncompliant").sum()
human_asr = human_success / human_total * 100 if human_total > 0 else 0

llm_total = len(llm_df)
llm_success = (llm_df["judge_label"] == "noncompliant").sum()
llm_asr = llm_success / llm_total * 100 if llm_total > 0 else 0

improvement = llm_asr - human_asr

# ==== ASR BY CATEGORY (overlapping categories only) ====
human_by_cat = (
    human_df.groupby("category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
)

llm_by_cat = (
    llm_df.groupby("category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
)

# Find overlapping categories
overlap_cats = set(human_by_cat.index) & set(llm_by_cat.index)

# ==== PLOTS ====

# 1. Overall ASR Comparison
fig, ax = plt.subplots(figsize=(6, 5))
methods = ["Human\nBaseline", "LLM\nGenerated"]
asrs = [human_asr, llm_asr]
colors = ["#3498db", "#e74c3c"]
bars = ax.bar(methods, asrs, color=colors, alpha=0.7, edgecolor="black")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
ax.set_title("Attack Success Rate: Human vs LLM-Generated", fontsize=14, fontweight='bold')
ax.set_ylim(0, max(asrs) * 1.2)
plt.tight_layout()
plt.savefig(OUT_DIR / "overall_asr_comparison.png", dpi=300)
plt.close()

# 2. Category-by-Category Comparison (overlapping only)
if overlap_cats:
    comparison_data = []
    for cat in sorted(overlap_cats):
        comparison_data.append({
            "Category": cat,
            "Human": human_by_cat[cat],
            "LLM": llm_by_cat[cat]
        })

    comp_df = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(comp_df))
    width = 0.35

    ax.bar([i - width/2 for i in x], comp_df["Human"], width,
           label="Human Baseline", color="#3498db", alpha=0.7)
    ax.bar([i + width/2 for i in x], comp_df["LLM"], width,
           label="LLM Generated", color="#e74c3c", alpha=0.7)

    ax.set_xlabel("Attack Category", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("ASR by Category: Human vs LLM", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Category"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "category_comparison.png", dpi=300)
    plt.close()

# ==== PRINT SUMMARY ====
print("\n" + "=" * 80)
print("HUMAN vs LLM BASELINE COMPARISON")
print("=" * 80)
print(f"\n📊 Overall Performance:")
print(f"   Human Baseline ASR:      {human_asr:.2f}% ({human_success}/{human_total})")
print(f"   LLM-Generated ASR:       {llm_asr:.2f}% ({llm_success}/{llm_total})")
print(f"   Improvement:             {improvement:+.2f} percentage points")
print(f"   Relative Improvement:    {(improvement/human_asr*100) if human_asr > 0 else 0:.1f}%")

print(f"\n📋 Category Overlap:")
print(f"   Human categories:        {len(human_by_cat)}")
print(f"   LLM categories:          {len(llm_by_cat)}")
print(f"   Overlapping categories:  {len(overlap_cats)}")

if overlap_cats:
    print(f"\n📈 ASR by Overlapping Category:")
    for cat in sorted(overlap_cats):
        h = human_by_cat[cat]
        l = llm_by_cat[cat]
        diff = l - h
        print(f"   {cat:30s} Human: {h:5.1f}%  LLM: {l:5.1f}%  (Δ {diff:+5.1f}%)")

# ==== WRITE MARKDOWN SUMMARY ====
with open(SUMMARY_PATH, "w") as f:
    f.write("# Human vs LLM Baseline Comparison\n\n")

    f.write("## Overall Performance\n\n")
    f.write(f"| Method | ASR | Success / Total |\n")
    f.write(f"|--------|-----|----------------|\n")
    f.write(f"| Human Baseline | {human_asr:.2f}% | {human_success}/{human_total} |\n")
    f.write(f"| LLM-Generated | {llm_asr:.2f}% | {llm_success}/{llm_total} |\n")
    f.write(f"| **Improvement** | **{improvement:+.2f}pp** | - |\n\n")

    f.write(f"**Relative Improvement:** {(improvement/human_asr*100) if human_asr > 0 else 0:.1f}%\n\n")

    if overlap_cats:
        f.write("## ASR by Category (Overlapping)\n\n")
        f.write("| Category | Human ASR | LLM ASR | Improvement |\n")
        f.write("|----------|-----------|---------|-------------|\n")
        for cat in sorted(overlap_cats):
            h = human_by_cat[cat]
            l = llm_by_cat[cat]
            diff = l - h
            f.write(f"| {cat} | {h:.1f}% | {l:.1f}% | {diff:+.1f}pp |\n")

    f.write("\n## Visualizations\n\n")
    f.write("![Overall ASR Comparison](figures/comparison/overall_asr_comparison.png)\n\n")
    if overlap_cats:
        f.write("![Category Comparison](figures/comparison/category_comparison.png)\n\n")

print(f"\n✅ Comparison complete!")
print(f"   Figures saved to: {OUT_DIR}")
print(f"   Summary written to: {SUMMARY_PATH}")
print("=" * 80)
