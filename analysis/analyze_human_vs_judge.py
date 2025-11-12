"""
analysis/analyze_human_vs_judge.py
----------------------------------
Compares human annotation results vs GPT-4 judge labels.
Computes compliance stats, agreement, and basic performance metrics.
Outputs Markdown summary + confusion matrix plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    cohen_kappa_score
)
from pathlib import Path

# ==== CONFIG ====
CSV_PATH = "annotations/human_baseline_labeled.csv"  # update if different
OUT_DIR = Path("analysis/figures")
SUMMARY_PATH = Path("analysis/human_vs_judge_summary.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(CSV_PATH)

# Handle missing / normalize text
df["judge_label"] = df["judge_label"].fillna("error").str.lower().str.strip()
# pick whichever label column exists
if "human_label" in df.columns:
    df["human_label"] = df["human_label"]
elif "annotator_1" in df.columns:
    df["human_label"] = df["annotator_1"]
elif "final_label" in df.columns:
    df["human_label"] = df["final_label"]
else:
    raise ValueError("No human label column found (expected 'human_label', 'annotator_1', or 'final_label').")

df["human_label"] = df["human_label"].fillna("uncertain").str.lower().str.strip()

# ==== HUMAN LABEL STATS ====
n_total = len(df)
human_counts = df["human_label"].value_counts()
judge_counts = df["judge_label"].value_counts()

human_compliant = (df["human_label"] == "compliant").sum()
human_noncompliant = (df["human_label"] == "noncompliant").sum()
human_asr = human_noncompliant / n_total * 100  # Attack Success Rate (human view)

print("===== HUMAN BASELINE STATS =====")
print(f"Total labeled: {n_total}")
print(f"Human Attack Success Rate (ASR): {human_asr:.1f}%")
print(human_counts)

# ==== COMPARE HUMAN vs JUDGE ====
valid_mask = df["human_label"].isin(["compliant", "noncompliant"]) & df["judge_label"].isin(["compliant", "noncompliant"])
df_valid = df[valid_mask].copy()

y_true = df_valid["human_label"]
y_pred = df_valid["judge_label"]

acc = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=3, output_dict=True)

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(y_true, y_pred, labels=["noncompliant", "compliant"])
cm_df = pd.DataFrame(
    cm,
    index=["Human: noncompliant", "Human: compliant"],
    columns=["Judge: noncompliant", "Judge: compliant"]
)

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Human vs GPT-4 Judge Decisions")
plt.tight_layout()
plt.savefig(OUT_DIR / "human_vs_judge_confusion.png", dpi=300)
plt.close()

# ==== PRINT SUMMARY ====
print("\n===== HUMAN vs JUDGE COMPARISON =====")
print(f"Agreement accuracy: {acc*100:.2f}%")
print(f"Cohen's κ: {kappa:.3f}")
print("Confusion Matrix:\n", cm_df)
print("\nClassification report:")
for k, v in report.items():
    if isinstance(v, dict) and k in ["compliant","noncompliant"]:
        print(f"{k}: Precision={v['precision']:.2f}, Recall={v['recall']:.2f}, F1={v['f1-score']:.2f}")

# ==== WRITE MARKDOWN SUMMARY ====
with open(SUMMARY_PATH, "w") as f:
    f.write("# Human vs GPT-4 Judge Summary\n\n")
    f.write(f"**Total labeled:** {n_total}\n\n")
    f.write(f"**Human Attack Success Rate (ASR):** {human_asr:.2f}%\n\n")
    f.write("## Human Label Breakdown\n")
    for label, count in human_counts.items():
        f.write(f"- **{label}**: {count}\n")

    f.write("\n## Judge Label Breakdown\n")
    for label, count in judge_counts.items():
        f.write(f"- **{label}**: {count}\n")

    f.write(f"\n**Agreement Accuracy:** {acc*100:.2f}%\n")
    f.write(f"**Cohen's κ:** {kappa:.3f}\n\n")

    f.write("## Classification Report\n")
    for k, v in report.items():
        if isinstance(v, dict) and k in ["compliant","noncompliant"]:
            f.write(f"- **{k}** → Precision={v['precision']:.2f}, Recall={v['recall']:.2f}, F1={v['f1-score']:.2f}\n")

    f.write("\n## Confusion Matrix\n")
    f.write("![Human vs GPT-4 Judge](figures/human_vs_judge_confusion.png)\n\n")

print(f"\nSummary written to {SUMMARY_PATH}")
