import pandas as pd

# Series: one labelled column
s = pd.Series([82, 91, 76], index=["Ali", "Sara", "Hamza"], name="marks")
print(s)
print(s["Sara"], s.mean(), s.max())

# DataFrame: a table
df = pd.DataFrame({
    "name": ["Ali", "Sara", "Hamza", "Zoya"],
    "dept": ["CS", "AI", "CS", "SE"],
    "marks": [82, 91, 76, 88],
})
print(df)
print(df.shape, list(df.columns))
print(df.dtypes)
print(df.head(2))
print(df.describe())

# columns and rows
print(df["name"])
print(df[["name", "marks"]])
print(df.loc[1])                    
print(df.iloc[0, 2])                

# new column
df["passed"] = df["marks"] >= 80
print(df)

# reading the real dataset
students = pd.read_csv("students.csv")
print(students.head())
students.info()
