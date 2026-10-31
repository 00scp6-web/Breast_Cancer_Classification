import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

os.makedirs("results", exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

original_features = X_train.columns.tolist()

variance_filter = VarianceThreshold(threshold=0)
X_train_filtered = pd.DataFrame(
    variance_filter.fit_transform(X_train),
    columns=X_train.columns[variance_filter.get_support()],
    index=X_train.index
)
X_test_filtered = X_test[X_train_filtered.columns]

correlation_matrix = X_train_filtered.corr().abs()
upper_triangle = correlation_matrix.where(
    pd.DataFrame(
        [[i < j for j in range(len(correlation_matrix.columns))]
         for i in range(len(correlation_matrix.columns))],
        index=correlation_matrix.index,
        columns=correlation_matrix.columns
    )
)
correlated_features = [
    column for column in upper_triangle.columns
    if (upper_triangle[column] > 0.95).any()
]

X_train_filtered = X_train_filtered.drop(columns=correlated_features)
X_test_filtered = X_test_filtered.drop(columns=correlated_features)

selector = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=min(10, X_train_filtered.shape[1])
)
X_train_selected = pd.DataFrame(
    selector.fit_transform(X_train_filtered, y_train),
    columns=X_train_filtered.columns[selector.get_support()],
    index=X_train.index
)
X_test_selected = X_test_filtered[X_train_selected.columns]

pd.DataFrame({
    "Selected Features": X_train_selected.columns
}).to_csv("results/selected_features.csv", index=False)

pd.DataFrame({
    "Stage": [
        "Original Features",
        "After Variance Filtering",
        "After Correlation Filtering",
        "After RFE"
    ],
    "Number of Features": [
        len(original_features),
        len(X_train_filtered.columns) + len(correlated_features),
        len(X_train_filtered.columns),
        len(X_train_selected.columns)
    ]
}).to_csv("results/feature_selection_summary.csv", index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": GaussianNB(),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


results = []

for feature_set_name, Xtr, Xte in [
    ("Original Features", X_train, X_test),
    ("Selected Features", X_train_selected, X_test_selected)
]:
    for model_name, model in models.items():
        model.fit(Xtr, y_train)
        predictions = model.predict(Xte)
        probabilities = model.predict_proba(Xte)[:, 1]

        results.append({
            "Feature Set": feature_set_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Malignant Recall": recall_score(
                y_test, predictions, pos_label=0, zero_division=0
            ),
            "Confusion Matrix": str(confusion_matrix(y_test, predictions).tolist())
        })

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nFEATURE SELECTION SUMMARY")
print(pd.read_csv("results/feature_selection_summary.csv").to_string(index=False))

print("\nSELECTED FEATURES")
print(", ".join(X_train_selected.columns))

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

for metric in ["Accuracy", "F1 Score", "ROC-AUC", "Malignant Recall"]:
    pivot = results_df.pivot(index="Model", columns="Feature Set", values=metric)
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title(f"{metric}: Original vs Selected Features")
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"results/{metric.lower().replace(' ', '_')}_comparison.png")
    plt.close()

print("\nResults saved in the 'results' folder.")
