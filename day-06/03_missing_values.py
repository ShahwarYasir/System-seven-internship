import pandas as pd

df = pd.read_csv("students.csv")

print(df.isna().sum())              # missing values per column
print(df.isna().mean().round(3))    # as a fraction
print(df[df["cgpa"].isna()].head(3))

# drop rows with any missing value
print("dropna:", df.dropna().shape)

# fill with a fixed value, mean, median or mode
filled = df.copy()
filled["study_hours"] = filled["study_hours"].fillna(filled["study_hours"].mean())
filled["attendance"] = filled["attendance"].fillna(filled["attendance"].median())
filled["department"] = filled["department"].fillna(filled["department"].mode()[0])
print(filled.isna().sum())

# forward fill / interpolate suit ordered (time-series) data
s = pd.Series([1.0, None, None, 4.0])
print(s.ffill().tolist())           # [1.0, 1.0, 1.0, 4.0]
print(s.interpolate().tolist())     # [1.0, 2.0, 3.0, 4.0]
