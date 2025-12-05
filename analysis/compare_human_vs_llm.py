"""
analysis/compare_human_vs_llm.py
---------------------------------
Comprehensive comparison of Human vs LLM-Generated baselines.

Compares:
1. Attack Performance: Human-crafted vs LLM-generated prompts
2. Judge Reliability: GPT-4 judge accuracy on both baselines

Required data:
- Human baseline: data/human_baseline_labeled.csv (with human_label column)
- LLM baseline: data/gpt_baseline_attack.csv (with human_label column if available)

Generates:
- Attack success rate comparison
- Judge reliability comparison (if human labels exist)
- Category-level analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

# ==== CONFIG ====
HUMAN_CSV = "data/human_baseline_labeled.csv"
LLM_CSV = "data/gpt_baseline_labeled.csv"
OUT_DIR = Path("analysis/figures/comparison")
SUMMARY_PATH = Path("analysis/human_vs_llm_comparison.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== LOAD DATA ====
print("📂 Loading human baseline...")
human_df = pd.read_csv(HUMAN_CSV)
human_df["judge_label"] = human_df["judge_label"].fillna("error").str.lower().str.strip()

print("📂 Loading LLM baseline...")
llm_df = pd.read_csv(LLM_CSV)
llm_df["judge_label"] = llm_df["judge_label"].fillna("error").str.lower().str.strip()

# Check for human labels
human_has_labels = "human_label" in human_df.columns
llm_has_labels = "human_label" in llm_df.columns

if human_has_labels:
    human_df["human_label"] = human_df["human_label"].fillna("").str.lower().str.strip()
if llm_has_labels:
    llm_df["human_label"] = llm_df["human_label"].fillna("").str.lower().str.strip()

print(f"   Human baseline: {len(human_df)} prompts, human labels: {human_has_labels}")
print(f"   LLM baseline: {len(llm_df)} prompts, human labels: {llm_has_labels}")

# ==== 1. ATTACK PERFORMANCE COMPARISON ====
print("\n" + "=" * 80)
print("PART 1: ATTACK METHOD COMPARISON")
print("=" * 80)

# Overall ASR (using GPT-4 judge)
human_total = len(human_df)
human_success = (human_df["judge_label"] == "noncompliant").sum()
human_asr_judge = human_success / human_total * 100 if human_total > 0 else 0

llm_total = len(llm_df)
llm_success = (llm_df["judge_label"] == "noncompliant").sum()
llm_asr_judge = llm_success / llm_total * 100 if llm_total > 0 else 0

improvement = llm_asr_judge - human_asr_judge

# ASR by human labels (if available)
if human_has_labels:
    human_success_human = (human_df["human_label"] == "noncompliant").sum()
    human_asr_human = human_success_human / human_total * 100 if human_total > 0 else 0
else:
    human_asr_human = None

if llm_has_labels:
    llm_success_human = (llm_df["human_label"] == "noncompliant").sum()
    llm_asr_human = llm_success_human / llm_total * 100 if llm_total > 0 else 0
else:
    llm_asr_human = None

print(f"\n📊 Attack Success Rates:")
print(f"   Human Baseline (Judge):  {human_asr_judge:.2f}% ({human_success}/{human_total})")
if human_asr_human is not None:
    print(f"   Human Baseline (Human):  {human_asr_human:.2f}% ({human_success_human}/{human_total})")
print(f"   LLM Baseline (Judge):    {llm_asr_judge:.2f}% ({llm_success}/{llm_total})")
if llm_asr_human is not None:
    print(f"   LLM Baseline (Human):    {llm_asr_human:.2f}% ({llm_success_human}/{llm_total})")
print(f"   Improvement (Judge):     {improvement:+.2f}pp")

# ASR by category
human_by_cat = (
    human_df.groupby("category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
)

llm_by_cat = (
    llm_df.groupby("category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
)

overlap_cats = set(human_by_cat.index) & set(llm_by_cat.index)

# ==== 2. JUDGE RELIABILITY COMPARISON ====
print("\n" + "=" * 80)
print("PART 2: JUDGE RELIABILITY COMPARISON")
print("=" * 80)

judge_stats = {}

# Human baseline judge reliability
if human_has_labels:
    human_valid = human_df[
        human_df["human_label"].isin(["compliant", "noncompliant"]) &
        human_df["judge_label"].isin(["compliant", "noncompliant"])
    ].copy()

    if len(human_valid) > 0:
        human_judge_acc = accuracy_score(human_valid["human_label"], human_valid["judge_label"])
        human_judge_kappa = cohen_kappa_score(human_valid["human_label"], human_valid["judge_label"])

        judge_stats["human"] = {
            "accuracy": human_judge_acc * 100,
            "kappa": human_judge_kappa,
            "n": len(human_valid)
        }

        print(f"\n🔍 Human Baseline Judge Reliability:")
        print(f"   Agreement:  {human_judge_acc*100:.2f}%")
        print(f"   Cohen's κ:  {human_judge_kappa:.3f}")
        print(f"   Samples:    {len(human_valid)}")
    else:
        print(f"\n⚠️ Human baseline: No valid labels for comparison")
else:
    print(f"\n⚠️ Human baseline: No human labels available")

# LLM baseline judge reliability
if llm_has_labels:
    llm_valid = llm_df[
        llm_df["human_label"].isin(["compliant", "noncompliant"]) &
        llm_df["judge_label"].isin(["compliant", "noncompliant"])
    ].copy()

    if len(llm_valid) > 0:
        llm_judge_acc = accuracy_score(llm_valid["human_label"], llm_valid["judge_label"])
        llm_judge_kappa = cohen_kappa_score(llm_valid["human_label"], llm_valid["judge_label"])

        judge_stats["llm"] = {
            "accuracy": llm_judge_acc * 100,
            "kappa": llm_judge_kappa,
            "n": len(llm_valid)
        }

        print(f"\n🔍 LLM Baseline Judge Reliability:")
        print(f"   Agreement:  {llm_judge_acc*100:.2f}%")
        print(f"   Cohen's κ:  {llm_judge_kappa:.3f}")
        print(f"   Samples:    {len(llm_valid)}")
    else:
        print(f"\n⚠️ LLM baseline: No valid labels for comparison")
else:
    print(f"\n⚠️ LLM baseline: No human labels available")

# ==== VISUALIZATIONS ====

# 1. Overall ASR Comparison (Judge-based)
fig, ax = plt.subplots(figsize=(7, 5))
methods = ["Human\nBaseline", "LLM\nGenerated"]
asrs = [human_asr_judge, llm_asr_judge]
colors = ["#3498db", "#e74c3c"]
bars = ax.bar(methods, asrs, color=colors, alpha=0.7, edgecolor="black", linewidth=2)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
ax.set_title("Attack Success Rate: Human vs LLM-Generated\n(GPT-4 Judge)", fontsize=14, fontweight='bold')
ax.set_ylim(0, max(asrs) * 1.3 if max(asrs) > 0 else 10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "overall_asr_comparison.png", dpi=300)
plt.close()

# 2. Judge Reliability Comparison (if both have human labels)
if "human" in judge_stats and "llm" in judge_stats:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    methods = ["Human\nBaseline", "LLM\nBaseline"]
    accuracies = [judge_stats["human"]["accuracy"], judge_stats["llm"]["accuracy"]]
    bars = ax1.bar(methods, accuracies, color=["#3498db", "#e74c3c"], alpha=0.7, edgecolor="black", linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax1.set_ylabel("Judge Agreement (%)", fontsize=12)
    ax1.set_title("Judge Accuracy\n(Human vs GPT-4)", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.legend()

    # Cohen's Kappa comparison
    kappas = [judge_stats["human"]["kappa"], judge_stats["llm"]["kappa"]]
    bars = ax2.bar(methods, kappas, color=["#3498db", "#e74c3c"], alpha=0.7, edgecolor="black", linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax2.set_ylabel("Cohen's κ", fontsize=12)
    ax2.set_title("Judge Reliability\n(Inter-rater Agreement)", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='0.6 threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "judge_reliability_comparison.png", dpi=300)
    plt.close()

# 3. Category comparison (if overlap exists)
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

# ==== MARKDOWN SUMMARY ====
with open(SUMMARY_PATH, "w") as f:
    f.write("# Human vs LLM Baseline Comparison\n\n")

    f.write("## 1. Attack Performance Comparison\n\n")
    f.write("### Overall Attack Success Rates\n\n")
    f.write("| Method | Judge ASR | Human ASR | Success/Total |\n")
    f.write("|--------|-----------|-----------|---------------|\n")
    f.write(f"| Human Baseline | {human_asr_judge:.2f}% | ")
    f.write(f"{human_asr_human:.2f}% | " if human_asr_human else "N/A | ")
    f.write(f"{human_success}/{human_total} |\n")
    f.write(f"| LLM-Generated | {llm_asr_judge:.2f}% | ")
    f.write(f"{llm_asr_human:.2f}% | " if llm_asr_human else "N/A | ")
    f.write(f"{llm_success}/{llm_total} |\n")
    f.write(f"| **Improvement** | **{improvement:+.2f}pp** | - | - |\n\n")

    f.write(f"**Relative Improvement:** {(improvement/human_asr_judge*100) if human_asr_judge > 0 else 0:.1f}%\n\n")

    if overlap_cats:
        f.write("### ASR by Category (Overlapping)\n\n")
        f.write("| Category | Human ASR | LLM ASR | Improvement |\n")
        f.write("|----------|-----------|---------|-------------|\n")
        for cat in sorted(overlap_cats):
            h = human_by_cat[cat]
            l = llm_by_cat[cat]
            diff = l - h
            f.write(f"| {cat} | {h:.1f}% | {l:.1f}% | {diff:+.1f}pp |\n")
        f.write("\n")

    f.write("## 2. Judge Reliability Comparison\n\n")

    if judge_stats:
        f.write("### Agreement with Human Annotators\n\n")
        f.write("| Baseline | Accuracy | Cohen's κ | Samples |\n")
        f.write("|----------|----------|-----------|----------|\n")

        if "human" in judge_stats:
            s = judge_stats["human"]
            f.write(f"| Human Baseline | {s['accuracy']:.2f}% | {s['kappa']:.3f} | {s['n']} |\n")

        if "llm" in judge_stats:
            s = judge_stats["llm"]
            f.write(f"| LLM Baseline | {s['accuracy']:.2f}% | {s['kappa']:.3f} | {s['n']} |\n")

        f.write("\n")

        if "human" in judge_stats and "llm" in judge_stats:
            acc_diff = judge_stats["llm"]["accuracy"] - judge_stats["human"]["accuracy"]
            kappa_diff = judge_stats["llm"]["kappa"] - judge_stats["human"]["kappa"]

            f.write("### Judge Performance Difference\n\n")
            f.write(f"- Accuracy difference: {acc_diff:+.2f}pp\n")
            f.write(f"- Kappa difference: {kappa_diff:+.3f}\n\n")

            if abs(acc_diff) < 5:
                f.write("**Interpretation:** Judge performs similarly on both baselines.\n\n")
            elif acc_diff > 0:
                f.write("**Interpretation:** Judge is more reliable on LLM-generated attacks.\n\n")
            else:
                f.write("**Interpretation:** Judge is more reliable on human-crafted attacks.\n\n")
    else:
        f.write("*No human labels available for judge comparison.*\n\n")

    f.write("## Visualizations\n\n")
    f.write("![Overall ASR Comparison](figures/comparison/overall_asr_comparison.png)\n\n")

    if "human" in judge_stats and "llm" in judge_stats:
        f.write("![Judge Reliability Comparison](figures/comparison/judge_reliability_comparison.png)\n\n")

    if overlap_cats:
        f.write("![Category Comparison](figures/comparison/category_comparison.png)\n\n")

# ==== CONSOLE SUMMARY ====
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✅ Comparison complete!")
print(f"   Figures saved to: {OUT_DIR}")
print(f"   Summary written to: {SUMMARY_PATH}")

if judge_stats:
    print(f"\n📊 Key Findings:")
    print(f"   Attack Performance: LLM {improvement:+.2f}pp better than Human")
    if "human" in judge_stats and "llm" in judge_stats:
        acc_diff = judge_stats["llm"]["accuracy"] - judge_stats["human"]["accuracy"]
        print(f"   Judge Reliability: {acc_diff:+.2f}pp difference (LLM vs Human baseline)")

print("=" * 80)
