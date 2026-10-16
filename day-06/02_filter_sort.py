import pandas as pd

df = pd.read_csv("students.csv")

high = df[df["cgpa"] > 3.5]
print("CGPA > 3.5:", len(high))

# several conditions: use & | ~ with brackets
ai_strong = df[(df["department"] == "AI") & (df["cgpa"] >= 3.0)]
print(ai_strong[["student_id", "department", "cgpa"]].head())

print(df[df["department"].isin(["CS", "AI"])].shape)

# sorting
print(df.sort_values("cgpa", ascending=False).head(5)[["student_id", "cgpa"]])
print(df.sort_values(["department", "cgpa"], ascending=[True, False]).head())

# group summaries
print(df.groupby("department")["cgpa"].mean().round(2))
print(df.groupby("gender")[["study_hours", "cgpa"]].mean().round(2))
print(df["department"].value_counts())

# query() as a readable alternative
print(df.query("study_hours > 6 and attendance > 90").shape)
