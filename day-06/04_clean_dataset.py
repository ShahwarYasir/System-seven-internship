# Task 6: Clean a dataset with missing values; plot before and after
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

raw = pd.read_csv("students.csv")
print("Before:", raw.shape, "| missing:", int(raw.isna().sum().sum()),
      "| duplicates:", int(raw.duplicated().sum()))

clean = raw.copy()
clean = clean.drop_duplicates()
clean = clean.dropna(subset=["cgpa"])                
for col in ["study_hours", "attendance", "sleep_hours"]:
    clean[col] = clean[col].fillna(clean[col].median())
clean["department"] = clean["department"].fillna("Unknown")

print("After: ", clean.shape, "| missing:", int(clean.isna().sum().sum()),
      "| duplicates:", int(clean.duplicated().sum()))
clean.to_csv("students_clean.csv", index=False)

# plots: before vs after
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

sns.heatmap(raw.isna(), cbar=False, yticklabels=False, ax=axes[0, 0])
axes[0, 0].set_title("Missing values BEFORE (light = missing)")

sns.heatmap(clean.isna(), cbar=False, yticklabels=False, ax=axes[0, 1])
axes[0, 1].set_title("Missing values AFTER")

sns.histplot(raw["study_hours"].dropna(), kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Study hours BEFORE")

sns.histplot(clean["study_hours"], kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Study hours AFTER")

plt.tight_layout()
plt.savefig("before_after.png", dpi=120)
plt.show()
