# System 7 Internship

Week 1: Python and Data Science Foundations

## Day 1: Python basics
- Syntax: comments, indentation, multi-line strings
- Variables and data types
- Operators (arithmetic, comparison, logical)
- Input/output with f-strings
- Mini project: GPA calculator

## Day 2: Control flow and functions
- Conditions (if / elif / else)
- Loops (for, while, break, continue) and range()
- Functions (parameters, defaults, return values)
- Mini projects: command-line calculator, number-guessing game

## Day 3: Data structures
- Lists, tuples, dictionaries, sets
- Comprehensions, slicing, unpacking, set operations
- Mini project: word-frequency counter and contact book

## Day 4: Files, exceptions, modules and OOP
- File handling (read, write, append), try/except/finally, custom exceptions
- Modules (standard library and my own)
- Classes, inheritance, `__str__`
- Task 3: student record manager saving to JSON
- Task 4: BankAccount / SavingsAccount classes in their own module

## Day 5: NumPy
- Arrays, shapes, indexing, slicing, boolean masks
- Vectorised math, broadcasting, matrix operations
- Task 5: operations on a sample 5x5 matrix

## Day 6: Pandas
- Series and DataFrames, filtering, sorting, groupby
- Handling missing values and duplicates
- Task 6: cleaned `students.csv`, plotted before and after

## Day 7: Visualization, TensorFlow intro and CSV analysis
- Matplotlib and Seaborn basic charts
- TensorFlow installation and tensors
- Task 1: CSV analysis with summary statistics and charts (`day-07/04_csv_analysis.py`)

Week 2: Mathematics and Machine Learning Basics

## Day 8: Machine learning foundations
- AI vs ML vs deep learning, types of learning
- Samples, features and target in a Pandas DataFrame
- Regression vs classification (`day-08-ml-foundations/ml_foundations.ipynb`)

## Day 9: Descriptive statistics
- Mean, median, mode, variance and standard deviation by hand, in NumPy and in Pandas (population vs sample)
- How outliers affect the mean but not the median
- Created the synthetic practice dataset `houses_raw.csv` with `generate_houses.py`

## Day 10: Linear algebra
- Vectors, matrices, dot product, matrix multiplication and shape rules in NumPy
- Linear model predictions as `X @ w + b`
- Linear regression solved with the normal equation

## Day 11: Probability, correlation and data preparation
- Task 1: dataset exploration and preparation (missing values, duplicates, inconsistent text, outliers)
- Probability and correlation matrix
- 70/15/15 train, validation and test split


## How to run
```
pip install numpy pandas matplotlib seaborn tensorflow
cd day-07
python 04_csv_analysis.py
```
