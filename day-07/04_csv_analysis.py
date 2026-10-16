# Task 1 (main Week 1 task): CSV analysis with charts and summary statistics
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)

# 1. Load and inspect
df = pd.read_csv("students.csv")
print("Shape:", df.shape)
print(df.dtypes, "\n")
print("Missing values:\n", df.isna().sum(), "\n")

# 2. Clean
df = df.drop_duplicates().dropna(subset=["cgpa"])
for col in ["study_hours", "attendance", "sleep_hours"]:
    df[col] = df[col].fillna(df[col].median())
df["department"] = df["department"].fillna("Unknown")
print("Rows after cleaning:", len(df), "\n")

# 3. Summary statistics
num_cols = ["study_hours", "attendance", "sleep_hours", "cgpa"]
summary = df[num_cols].describe().round(2)
print(summary, "\n")

by_dept = df.groupby("department")["cgpa"].agg(["count", "mean", "median", "std"]).round(2)
print(by_dept, "\n")

corr = df[num_cols].corr().round(2)
print("Correlation with CGPA:\n", corr["cgpa"].drop("cgpa"), "\n")

top = df.nlargest(5, "cgpa")[["student_id", "department", "study_hours", "cgpa"]]
print("Top 5 students:\n", top, "\n")

# 4. Charts
plt.figure(figsize=(7, 5))
sns.histplot(df["cgpa"], bins=15, kde=True)
plt.title("CGPA distribution")
plt.savefig("charts/cgpa_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="department", y="cgpa")
plt.title("CGPA by department")
plt.savefig("charts/cgpa_by_department.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
sns.regplot(data=df, x="study_hours", y="cgpa", scatter_kws={"alpha": 0.5})
plt.title("Study hours vs CGPA")
plt.savefig("charts/study_vs_cgpa.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.savefig("charts/correlation.png", dpi=120, bbox_inches="tight")
plt.close()

# 5. Save the summary for the README / report
summary.to_csv("summary_statistics.csv")
with open("findings.txt", "w") as f:
    f.write(f"Students analysed: {len(df)}\n")
    f.write(f"Average CGPA: {df['cgpa'].mean():.2f}\n")
    known = by_dept.drop(index="Unknown", errors="ignore")
    f.write(f"Best department (mean CGPA): {known['mean'].idxmax()}\n")
    f.write(f"Correlation study_hours vs cgpa: {corr.loc['study_hours', 'cgpa']}\n")
print("Charts saved in charts/, summary in summary_statistics.csv and findings.txt")
