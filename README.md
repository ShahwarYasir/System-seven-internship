# System 7 Internship

Daily work for the System 7 remote AI/ML internship, committed one folder per day.

## Week 1: Python and Data Science Foundations

| Day | Topic | What I covered |
|---|---|---|
| Day 1 | Python basics | • Syntax: comments, indentation and multi-line strings<br>• Variables and data types<br>• Operators (arithmetic, comparison, logical)<br>• Input/output with f-strings<br>• Mini project: GPA calculator |
| Day 2 | Control flow and functions | • Conditions (if / elif / else)<br>• Loops (for, while, break, continue) and `range()`<br>• Functions: parameters, defaults and return values<br>• Mini projects: command-line calculator and number-guessing game |
| Day 3 | Data structures | • Lists, tuples, dictionaries and sets<br>• Comprehensions, slicing, unpacking and set operations<br>• Mini project: word-frequency counter and contact book |
| Day 4 | Files, exceptions, modules and OOP | • File handling (read, write, append) and try/except/finally, including custom exceptions<br>• Modules: standard library and my own<br>• Classes, inheritance and `__str__`<br>• Task 3: student record manager saving to JSON<br>• Task 4: `BankAccount` / `SavingsAccount` classes in their own module |
| Day 5 | NumPy | • Arrays, shapes, indexing, slicing and boolean masks<br>• Vectorised math, broadcasting and matrix operations<br>• Task 5: operations on a sample 5x5 matrix |
| Day 6 | Pandas | • Series and DataFrames, filtering, sorting and groupby<br>• Handling missing values and duplicates<br>• Task 6: cleaned `students.csv` and plotted it before and after |
| Day 7 | Visualization, TensorFlow intro and CSV analysis | • Matplotlib and Seaborn basic charts<br>• TensorFlow installation and tensors<br>• Task 1: CSV analysis with summary statistics and charts (`day-07/04_csv_analysis.py`) |

## Week 2: Mathematics and Machine Learning Basics

| Day | Topic | What I covered |
|---|---|---|
| Day 8 | Machine learning foundations | • AI vs ML vs deep learning and the types of learning<br>• Samples, features and target in a Pandas DataFrame<br>• Regression vs classification (`day-08-ml-foundations/ml_foundations.ipynb`) |
| Day 9 | Descriptive statistics | • Mean, median, mode, variance and standard deviation by hand, in NumPy and in Pandas (population vs sample)<br>• How outliers affect the mean but not the median<br>• Created the synthetic practice dataset `houses_raw.csv` with `generate_houses.py` |
| Day 10 | Linear algebra | • Vectors, matrices, dot product, matrix multiplication and shape rules in NumPy<br>• Linear model predictions as `X @ w + b`<br>• Linear regression solved with the normal equation |
| Day 11 | Probability, correlation and data preparation | • Task 1: dataset exploration and preparation (missing values, duplicates, inconsistent text, outliers)<br>• Probability and the correlation matrix<br>• 70/15/15 train, validation and test split |
| Day 12 | Regression with Scikit-learn | • Task 2: simple and multiple linear regression on house prices<br>• MAE, RMSE and R2, and how to read the coefficients<br>• Predicting the price of a new house |
| Day 13 | Classification models | • Task 3: logistic regression, KNN, decision tree and random forest<br>• Tuned k and `max_depth` on the validation set<br>• Compared the models with accuracy and the confusion matrix |
| Day 14 | Final models and feature importance | • Final regression and classification models scored once on the test set<br>• Coefficients and random forest feature importance<br>• Findings on what drives house prices |

## Week 3: Model Evaluation, Advanced ML and Deep Learning

| Day | Topic | What I covered |
|---|---|---|
| Day 15 | Model evaluation (`day-15-model-evaluation`) | • Q1: 80/20 stratified train-test split with a fixed `random_state`, checking shapes and class balance in both parts<br>• Q2: accuracy, precision, recall and F1 for logistic regression, plus a confusion-matrix plot<br>• Q3: MAE, MSE, RMSE and R2 for a house-price model, written by hand in NumPy and verified against scikit-learn<br>• Q4: 5-fold stratified cross-validation on logistic regression, KNN and random forest (mean and std)<br>• Q5: ROC curves and AUC for logistic regression vs random forest on one chart<br>• Learning note: choosing the right metric, and why accuracy misleads on imbalanced data |
| Day 16 | Tuning and model comparison (`day-16-tuning-model-comparison`) | • Q6: decision tree with `max_depth` 1 to 15, plotting training vs test score to show overfitting<br>• Q7: `ColumnTransformer` that scales numeric columns and one-hot encodes categorical columns in one step<br>• Q8: SVM tuned with `GridSearchCV` and `RandomizedSearchCV`, comparing best score and run time<br>• Practical task: five models tuned with `GridSearchCV` and compared on CV F1, precision, recall and ROC-AUC, with logistic regression chosen as the best and justified<br>• Learning note: the bias-variance trade-off |

## How to run

```
pip install numpy pandas scipy scikit-learn matplotlib seaborn tensorflow
cd day-07
python 04_csv_analysis.py
```

Week 2 notebooks:

```
cd day-14-house-price-classifier
jupyter notebook final_house_price_models.ipynb
```

Week 3 scripts (create the `outputs` folder first, since the scripts save plots there):

```
cd day-15-model-evaluation
mkdir -p outputs
python q2_classification_metrics.py

cd ../day-16-tuning-model-comparison
mkdir -p outputs
python model_comparison.py
```