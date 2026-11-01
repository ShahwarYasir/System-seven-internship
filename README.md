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
| Day 17 | SVM, Naive Bayes and clustering (`day-17-svm-clustering`) | • Q9: K-Means for k = 2 to 10, using the elbow curve and silhouette score to choose k<br>• Q10: Naive Bayes vs SVM on the same split, comparing precision and recall and explaining the gap<br>• Bonus: K-Means vs hierarchical clustering vs DBSCAN on round blobs and two moons, scored with the adjusted Rand index, plus a dendrogram<br>• Learning note: choosing k, and when DBSCAN is the better choice |
| Day 18 | PCA, imbalanced data, leakage and pipelines (`day-18-pca-pipelines`) | • Q11: handling a 95/5 imbalanced dataset with `class_weight`, comparing recall before and after<br>• Q12: PCA down to 2 components with a cluster plot and explained variance<br>• Q13: a scaler, PCA and classifier pipeline, plus data-leakage demos (scaling before the split, and feature selection on pure noise)<br>• Practical task: customer segmentation with engineered features, PCA and K-Means, with named segment profiles<br>• Learning notes: data leakage and the curse of dimensionality |
| Day 19 | Neural networks (`day-19-neural-networks`) | • Q14: a single perceptron written in NumPy that learns the AND gate<br>• Q15: a Keras Sequential model on tabular data with loss curves, comparing SGD and Adam<br>• Practical task: a neural network that predicts student pass/fail, compared against a logistic-regression baseline<br>• Demos of activation functions and of learning rate, batch size and epochs<br>• Learning notes: activation functions, and learning rate / batch size / epochs |
| Day 20 | CNN basics (`day-20-cnn-basics`) | • Q16: resizing to 64x64, normalizing, and 9 augmented samples using Keras augmentation layers<br>• Q17: a CNN with Conv2D and MaxPooling layers trained on CIFAR-10 (test accuracy 0.7693), with a confusion matrix and per-class analysis<br>• Bonus: a dense network vs the CNN, showing the parameter and accuracy gap<br>• Learning note: why CNNs suit images |
| Day 21 | Image classifier (`day-21-image-classifier`) | • Q18: MobileNetV2 transfer learning with a frozen base, compared with the Day 20 CNN on the same test images (test accuracy 0.8625 vs 0.7693), with prediction visualizations and the most confident mistakes<br>• Bonus: fine-tuning the top layers of MobileNetV2 and comparing it with feature extraction (0.8883 after fine-tuning)<br>• Practical task: the image classifier deliverable, including `predict_image.py` to classify any photo<br>• Learning note: transfer learning, feature extraction vs fine-tuning |

### Week 3 deliverables

| Deliverable | Where |
|---|---|
| Model comparison with a justified best model | `day-16-tuning-model-comparison/model_comparison.py` |
| Segmentation analysis | `day-18-pca-pipelines/segmentation_task.py` |
| Neural network (student results) | `day-19-neural-networks/student_results_nn.py` |
| Image classifier | `day-21-image-classifier/` |

## How to run

```
pip install numpy pandas scipy scikit-learn matplotlib seaborn tensorflow pillow
cd day-07
python 04_csv_analysis.py
```

Week 2 notebooks:

```
cd day-14-house-price-classifier
jupyter notebook final_house_price_models.ipynb
```

Week 3 scripts (each script creates its own `outputs` folder). Days 20 and 21 train image models, so use a GPU such as Colab for those:

```
cd day-15-model-evaluation
python q2_classification_metrics.py

cd ../day-16-tuning-model-comparison
python model_comparison.py

cd ../day-20-cnn-basics
python q17_cnn_cifar10.py

cd ../day-21-image-classifier
python q18_transfer_learning_mobilenetv2.py
```
