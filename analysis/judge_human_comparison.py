import pandas as pd

df = pd.read_csv("annotations/human_baseline_labeled.csv")
match = (df["judge_label"] == df["annotator_1"]).mean() * 100
print(f"Judge accuracy vs human: {match:.1f}%")
