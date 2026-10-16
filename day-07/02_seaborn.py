import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("students.csv").dropna()

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

sns.countplot(data=df, x="department", ax=axes[0, 0])
axes[0, 0].set_title("Students per department")

sns.boxplot(data=df, x="department", y="cgpa", ax=axes[0, 1])
axes[0, 1].set_title("CGPA by department")

sns.scatterplot(data=df, x="study_hours", y="cgpa", hue="gender", ax=axes[1, 0])
axes[1, 0].set_title("Study hours vs CGPA")

sns.heatmap(df.select_dtypes("number").drop(columns="student_id").corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Correlation matrix")

plt.tight_layout()
plt.savefig("seaborn_basics.png", dpi=120)
plt.show()
