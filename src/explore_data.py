from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt

from data_validation import load_data

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = load_data()

# 1. Which rows are duplicates, and what classes are they?
dupes = df[df.duplicated(keep="first")]
print("Duplicate rows by class (rows that would be removed):")
print(dupes["Class"].value_counts())

# 2. Class balance chart
counts = df["Class"].value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["steelblue", "crimson"])
plt.yscale("log")  # log scale so the tiny fraud bar is visible
plt.ylabel("Number of transactions (log scale)")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
plt.close()

# 3. Transaction amount by class
print("\nAmount summary by class:")
print(df.groupby("Class")["Amount"].describe().round(2))

plt.figure(figsize=(7, 4))
plt.hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.6, label="Legitimate", log=True)
plt.hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.6, label="Fraud", log=True)
plt.xlabel("Amount")
plt.ylabel("Count (log scale)")
plt.title("Transaction amount by class")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "amount_by_class.png", dpi=150)
plt.close()

print(f"\nFigures saved to {FIG_DIR}")