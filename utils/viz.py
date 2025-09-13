# utils/viz.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


sns.set(context='talk', style='whitegrid')


def pairplot(df, cols, out_path):
p = sns.pairplot(df[cols], corner=True)
out_path = Path(out_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
p.savefig(out_path, bbox_inches='tight')


def corr_heatmap(df, out_path):
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', center=0)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(out_path)
