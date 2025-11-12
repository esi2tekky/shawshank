"""
analysis/analyze_llm_baseline.py
---------------------------------
Comprehensive analysis for LLM-generated adversarial prompts.

Analyzes: data/gpt_baseline_attack.csv

Generates:
- Attack Success Rate (ASR) by evasion category
- ASR by intent category
- Compliance vs Noncompliance breakdown
- Latency histogram
- Markdown summary
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==== CONFIG ====
CSV_PATH = "data/gpt_baseline_attack.csv"
OUT_DIR = Path("analysis/figures/llm_baseline")
SUMMARY_PATH = Path("analysis/llm_baseline_summary.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== LOAD DATA ====
print(f"📂 Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Normalize labels
df["judge_label"] = df["judge_label"].fillna("error").str.lower()

# Parse latency from target_metadata
if "target_metadata" in df.columns:
    df["latency_s"] = df["target_metadata"].astype(str).str.extract(r"'latency_s': ([0-9.]+)").astype(float)
else:
    df["latency_s"] = None

# ==== COMPUTE METRICS ====
total = len(df)
n_success = (df["judge_label"] == "noncompliant").sum()
n_compliant = (df["judge_label"] == "compliant").sum()
n_uncertain = (df["judge_label"] == "uncertain").sum()
n_error = (df["judge_label"] == "error").sum()

asr = n_success / total * 100 if total > 0 else 0
judge_error_rate = n_error / total * 100 if total > 0 else 0

# ASR by evasion strategy (category)
by_evasion = (
    df.groupby("category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
    .sort_values(ascending=False)
)

# ASR by intent category
by_intent = (
    df.groupby("intent_category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
    .sort_values(ascending=False)
)

# ==== PLOTS ====

# 1. ASR by evasion strategy
plt.figure(figsize=(8, 5))
sns.barplot(x=by_evasion.values, y=by_evasion.index, palette="Blues_d")
plt.xlabel("Attack Success Rate (%)")
plt.ylabel("Evasion Strategy")
plt.title("ASR by Evasion Strategy (LLM-Generated)")
plt.tight_layout()
plt.savefig(OUT_DIR / "asr_by_evasion_strategy.png", dpi=300)
plt.close()

# 2. ASR by intent category
plt.figure(figsize=(8, 6))
sns.barplot(x=by_intent.values, y=by_intent.index, palette="Reds_d")
plt.xlabel("Attack Success Rate (%)")
plt.ylabel("Intent Category")
plt.title("ASR by Intent Category (LLM-Generated)")
plt.tight_layout()
plt.savefig(OUT_DIR / "asr_by_intent_category.png", dpi=300)
plt.close()

# 3. Compliance pie chart
labels = df["judge_label"].value_counts().index
sizes = df["judge_label"].value_counts().values
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Response Breakdown (LLM-Generated Prompts)")
plt.tight_layout()
plt.savefig(OUT_DIR / "compliance_pie.png", dpi=300)
plt.close()

# 4. Latency histogram (if available)
if df["latency_s"].notnull().any():
    plt.figure(figsize=(7, 4))
    sns.histplot(df["latency_s"].dropna(), bins=20, kde=True, color="slateblue")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Count")
    plt.title("Response Latency Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latency_hist.png", dpi=300)
    plt.close()

# ==== PRINT SUMMARY ====
print("\n" + "=" * 80)
print("LLM BASELINE ANALYSIS")
print("=" * 80)
print(f"Total prompts: {total}")
print(f"Attack Success Rate (ASR): {asr:.2f}%")
print(f"  - Compliant: {n_compliant}")
print(f"  - Noncompliant: {n_success}")
print(f"  - Uncertain: {n_uncertain}")
print(f"  - Error: {n_error}")
print(f"Judge Error Rate: {judge_error_rate:.2f}%")
print()
print("ASR by Evasion Strategy:")
for strategy, val in by_evasion.items():
    print(f"  - {strategy}: {val:.2f}%")
print()
print("ASR by Intent Category:")
for intent, val in by_intent.items():
    print(f"  - {intent}: {val:.2f}%")

# ==== WRITE MARKDOWN SUMMARY ====
with open(SUMMARY_PATH, "w") as f:
    f.write("# LLM Baseline Analysis\n\n")
    f.write(f"**Total prompts:** {total}\n\n")
    f.write(f"**Attack Success Rate (ASR):** {asr:.2f}%\n\n")
    f.write(f"- Compliant: {n_compliant}\n")
    f.write(f"- Noncompliant: {n_success}\n")
    f.write(f"- Uncertain: {n_uncertain}\n")
    f.write(f"- Error: {n_error}\n\n")
    f.write(f"**Judge Error Rate:** {judge_error_rate:.2f}%\n\n")

    f.write("## ASR by Evasion Strategy\n")
    for strategy, val in by_evasion.items():
        f.write(f"- **{strategy}**: {val:.2f}%\n")

    f.write("\n## ASR by Intent Category\n")
    for intent, val in by_intent.items():
        f.write(f"- **{intent}**: {val:.2f}%\n")

    f.write("\n## Figures\n")
    f.write("![ASR by Evasion Strategy](figures/llm_baseline/asr_by_evasion_strategy.png)\n\n")
    f.write("![ASR by Intent Category](figures/llm_baseline/asr_by_intent_category.png)\n\n")
    f.write("![Compliance Pie](figures/llm_baseline/compliance_pie.png)\n\n")
    if df["latency_s"].notnull().any():
        f.write("![Latency Histogram](figures/llm_baseline/latency_hist.png)\n\n")

print(f"\n✅ Analysis complete!")
print(f"   Figures saved to: {OUT_DIR}")
print(f"   Summary written to: {SUMMARY_PATH}")
print("=" * 80)
