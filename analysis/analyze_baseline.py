"""
analysis/analyze_baseline.py
----------------------------
Comprehensive baseline analysis for LLM jailbreak experiments.

Generates:
- Attack Success Rate (ASR) by category
- Compliance vs Noncompliance pie chart
- Judge error counts
- Latency histogram (if available)
- Markdown summary file with metrics
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==== CONFIG ====
CSV_PATH = "analysis/human_baseline_results.csv"
OUT_DIR = Path("analysis/figures")
SUMMARY_PATH = Path("analysis/summary.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(CSV_PATH)

# Normalize labels
df["judge_label"] = df["judge_label"].fillna("error").str.lower()

# Try to parse latency if exists
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

asr = n_success / total * 100
judge_error_rate = n_error / total * 100

by_cat = (
    df.groupby("category")["judge_label"]
    .apply(lambda s: (s == "noncompliant").mean() * 100)
    .sort_values(ascending=False)
)

# ==== PLOTS ====

# 1. ASR by category
plt.figure(figsize=(6,4))
sns.barplot(x=by_cat.values, y=by_cat.index, palette="Blues_d")
plt.xlabel("Attack Success Rate (%)")
plt.ylabel("Category")
plt.title("ASR by Prompt Category")
plt.tight_layout()
plt.savefig(OUT_DIR / "asr_by_category.png", dpi=300)
plt.close()

# 2. Compliance vs Noncompliance pie
labels = df["judge_label"].value_counts().index
sizes = df["judge_label"].value_counts().values
plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Response Breakdown by Judge Decision")
plt.tight_layout()
plt.savefig(OUT_DIR / "compliance_pie.png", dpi=300)
plt.close()

# 3. Latency histogram (if available)
if df["latency_s"].notnull().any():
    plt.figure(figsize=(6,4))
    sns.histplot(df["latency_s"].dropna(), bins=15, kde=True, color="slateblue")
    plt.xlabel("Latency (seconds)")
    plt.title("Response Latency Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latency_hist.png", dpi=300)
    plt.close()

# ==== PRINT SUMMARY TO CONSOLE ====
print("\n===== Baseline Summary =====")
print(f"Total prompts: {total}")
print(f"Attack Success Rate (ASR): {asr:.2f}%")
print(f"Compliant: {n_compliant} | Noncompliant: {n_success} | Uncertain: {n_uncertain} | Error: {n_error}")
print(f"Judge Error Rate: {judge_error_rate:.2f}%")
print("\nASR by category:")
print(by_cat)

# ==== WRITE MARKDOWN SUMMARY ====
with open(SUMMARY_PATH, "w") as f:
    f.write("# Baseline Summary\n\n")
    f.write(f"**Total prompts:** {total}\n\n")
    f.write(f"**Attack Success Rate (ASR):** {asr:.2f}%\n\n")
    f.write(f"- Compliant: {n_compliant}\n")
    f.write(f"- Noncompliant: {n_success}\n")
    f.write(f"- Uncertain: {n_uncertain}\n")
    f.write(f"- Error: {n_error}\n\n")
    f.write(f"**Judge Error Rate:** {judge_error_rate:.2f}%\n\n")

    f.write("## ASR by Category\n")
    for cat, val in by_cat.items():
        f.write(f"- **{cat}**: {val:.2f}%\n")

    f.write("\n## Figures\n")
    f.write("![ASR by Category](figures/asr_by_category.png)\n\n")
    f.write("![Compliance Pie](figures/compliance_pie.png)\n\n")
    if df["latency_s"].notnull().any():
        f.write("![Latency Histogram](figures/latency_hist.png)\n\n")

print(f"\n✅ Analysis complete. Figures saved to {OUT_DIR}, summary written to {SUMMARY_PATH}")
