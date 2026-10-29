"""Q10 (Medium): Train Naive Bayes and an SVM on the same data and compare
precision and recall."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

models = {
    "Naive Bayes": GaussianNB(),   # no scaling needed
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": model.score(X_test, y_test),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
    })
    print(f"--- {name}")
    print(classification_report(y_test, pred, target_names=data.target_names, digits=3))

print(pd.DataFrame(rows).round(4).to_string(index=False))
