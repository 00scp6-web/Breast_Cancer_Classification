from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


st.markdown(
    """
    <style>
    :root {
        --ink: #172554;
        --muted: #64748b;
        --line: #e2e8f0;
        --background: #f3f6fb;
    }

    .stApp {
        background: var(--background);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554 0%, #263b70 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 5px;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 2.6rem;
    }

    h1, h2, h3 {
        color: #172554;
        letter-spacing: -0.025em;
    }

    .hero {
        padding: 32px 34px;
        border-radius: 22px;
        background: linear-gradient(
            115deg,
            #1d4ed8 0%,
            #4f46e5 55%,
            #7c3aed 100%
        );
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #eef2ff !important;
        margin: 5px 0;
        max-width: 850px;
    }

    .eyebrow {
        color: #dbeafe;
        text-transform: uppercase;
        font-size: 0.76rem;
        letter-spacing: 0.13em;
        font-weight: 750;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px 18px;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .small-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
    }

    .small-card h4 {
        color: #1e40af;
        margin: 0 0 7px 0;
    }

    .small-card p {
        color: #475569;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_saved_tables():
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    summary_path = RESULTS_DIR / "feature_selection_summary.csv"
    selected_path = RESULTS_DIR / "selected_features.csv"

    missing = [
        path.name
        for path in [
            comparison_path,
            summary_path,
            selected_path,
        ]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing file(s) in results folder: "
            + ", ".join(missing)
        )

    comparison = pd.read_csv(comparison_path)
    summary = pd.read_csv(summary_path)
    selected_table = pd.read_csv(selected_path)

    return comparison, summary, selected_table


def get_selected_feature_names(selected_table, all_features):
    possible_columns = [
        column
        for column in selected_table.columns
        if column.strip().lower()
        in {"feature", "selected feature", "features"}
    ]

    column = (
        possible_columns[0]
        if possible_columns
        else selected_table.columns[0]
    )

    names = (
        selected_table[column]
        .dropna()
        .astype(str)
        .tolist()
    )

    names = [
        name for name in names
        if name in all_features
    ]

    if not names:
        raise ValueError(
            "Could not identify feature names in "
            "results/selected_features.csv."
        )

    return names


def convert_to_percentage(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_values = values.dropna()

    if not valid_values.empty and valid_values.max() <= 1:
        values = values * 100

    return values


def validate_comparison_data(comparison):
    required_columns = [
        "Feature Set",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Malignant Recall",
    ]

    missing = [
        column
        for column in required_columns
        if column not in comparison.columns
    ]

    if missing:
        raise ValueError(
            "model_comparison.csv is missing these columns: "
            + ", ".join(missing)
        )

    comparison = comparison.copy()

    comparison["Feature Set"] = (
        comparison["Feature Set"]
        .astype(str)
        .str.strip()
    )

    comparison["Model"] = (
        comparison["Model"]
        .astype(str)
        .str.strip()
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Malignant Recall",
    ]

    for column in metric_columns:
        comparison[column] = pd.to_numeric(
            comparison[column],
            errors="coerce",
        )

    return comparison


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5,
                ),
            ),
        ]),

        "Naive Bayes": GaussianNB(),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    probability=True,
                    random_state=42,
                ),
            ),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),

        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                random_state=42,
            ),
            n_estimators=100,
            random_state=42,
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


@st.cache_resource
def load_and_train_models():
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )

    y = pd.Series(
        dataset.target,
        name="target",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    (
        comparison,
        feature_summary,
        selected_table,
    ) = load_saved_tables()

    comparison = validate_comparison_data(comparison)

    selected_features = get_selected_feature_names(
        selected_table,
        X.columns,
    )

    variance_filter = VarianceThreshold(
        threshold=0,
    )

    X_train_var = pd.DataFrame(
        variance_filter.fit_transform(X_train),
        columns=X_train.columns[
            variance_filter.get_support()
        ],
        index=X_train.index,
    )

    correlation = X_train_var.corr().abs()

    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1,
        ).astype(bool)
    )

    correlated = [
        column
        for column in upper.columns
        if (upper[column] > 0.95).any()
    ]

    remaining_after_correlation = [
        column
        for column in X_train_var.columns
        if column not in correlated
    ]

    feature_sets = {
        "Original Features": list(X.columns),
        "Selected Features": selected_features,
    }

    trained = {}

    for feature_set_name, columns in feature_sets.items():
        Xtr = X_train[columns]
        Xte = X_test[columns]

        trained[feature_set_name] = {}

        for name, model in build_models().items():
            model.fit(Xtr, y_train)

            predictions = model.predict(Xte)
            probabilities = model.predict_proba(Xte)[:, 1]

            trained[feature_set_name][name] = {
                "model": model,
                "predictions": predictions,
                "probabilities": probabilities,

                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),

                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),

                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),

                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),

                "roc_auc": roc_auc_score(
                    y_test,
                    probabilities,
                ),

                "malignant_recall": recall_score(
                    y_test,
                    predictions,
                    pos_label=0,
                    zero_division=0,
                ),

                "confusion_matrix": confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1],
                ),
            }

    return (
        dataset,
        X,
        X_train,
        X_test,
        y_train,
        y_test,
        comparison,
        feature_summary,
        selected_features,
        remaining_after_correlation,
        trained,
    )


try:
    (
        data,
        X,
        X_train,
        X_test,
        y_train,
        y_test,
        comparison_df,
        feature_summary_df,
        selected_features,
        remaining_after_correlation,
        trained_models,
    ) = load_and_train_models()

except Exception as error:
    st.error(
        f"The dashboard could not load the project results: {error}"
    )

    st.info(
        "Check that model_comparison.csv, "
        "feature_selection_summary.csv, and "
        "selected_features.csv are inside the results folder."
    )

    st.stop()


st.sidebar.markdown("## 🔬 Model Studio")
st.sidebar.caption("BREAST CANCER CLASSIFICATION")
st.sidebar.markdown("---")

pages = [
    "Overview",
    "Model Arena",
    "Prediction Lab",
    "Performance Analysis",
    "Feature Importance",
]

page = st.sidebar.radio(
    "WORKSPACE",
    pages,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Machine Learning Mini Project**")
st.sidebar.caption("Academic Year 2026–27")
st.sidebar.caption(
    "Educational use only — not a medical diagnostic tool."
)


feature_set = "Original Features"

if page in [
    "Model Arena",
    "Prediction Lab",
    "Performance Analysis",
    "Feature Importance",
]:
    feature_set = st.sidebar.radio(
        "FEATURE SET",
        [
            "Original Features",
            "Selected Features",
        ],
        index=0,
    )


active_models = trained_models[feature_set]

active_columns = (
    X.columns.tolist()
    if feature_set == "Original Features"
    else selected_features
)

saved_active = comparison_df[
    comparison_df["Feature Set"] == feature_set
].copy()

if saved_active.empty:
    st.warning(
        f"No saved results found for {feature_set}. "
        "Check the Feature Set values in model_comparison.csv."
    )
    st.stop()


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">
            Machine Learning · Comparative Study
          </div>
          <h1>Breast Cancer Classification</h1>
          <p>
            A comparative study of classification and ensemble
            learning techniques.
          </p>
          <p>
            Compare nine algorithms and examine the effect
            of feature selection on model performance.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Educational project only. This application is not "
        "intended for medical diagnosis or treatment decisions."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset samples",
        f"{len(X):,}",
    )

    c2.metric(
        "Original features",
        f"{X.shape[1]}",
    )

    c3.metric(
        "Selected features",
        f"{len(selected_features)}",
    )

    c4.metric(
        "Algorithms",
        "9",
    )

    st.subheader("Algorithms evaluated")

    st.dataframe(
        pd.DataFrame({
            "Algorithm": list(build_models().keys()),

            "Category": [
                "Linear classifier",
                "Tree-based",
                "Instance-based",
                "Probabilistic",
                "Kernel-based",
                "Bagging ensemble",
                "Bagging ensemble",
                "Boosting ensemble",
                "Boosting ensemble",
            ],
        }),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Feature-selection summary")

    st.dataframe(
        feature_summary_df,
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Dataset details")

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="small-card">
                <h4>Wisconsin Diagnostic Dataset</h4>
                <p>
                    569 samples with 30 numerical features
                    computed from digitized images of
                    fine needle aspirates.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="small-card">
                <h4>Target classes</h4>
                <p>
                    0 = Malignant · 1 = Benign<br>
                    Stratified 80:20 training and test split.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# MODEL ARENA
# --------------------------------------------------

elif page == "Model Arena":

    st.title("🏆 Model Arena")

    st.caption(
        f"Comparing all nine algorithms using "
        f"{len(active_columns)} features ({feature_set.lower()})."
    )

    metric_map = {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1-score": "F1 Score",
        "ROC-AUC": "ROC-AUC",
        "Malignant recall": "Malignant Recall",
    }

    metric_label = st.selectbox(
        "Choose a metric",
        list(metric_map.keys()),
    )

    metric_col = metric_map[metric_label]
    chart_col = f"{metric_label} (%)"

    display_df = saved_active[
        ["Model", metric_col]
    ].copy()

    display_df[chart_col] = convert_to_percentage(
        display_df.pop(metric_col)
    )

    display_df = display_df.dropna(
        subset=[chart_col]
    ).sort_values(
        chart_col,
        ascending=False,
    )

    st.dataframe(
        display_df.round(2),
        hide_index=True,
        width="stretch",
    )

    st.subheader(f"{metric_label} Comparison")

    st.bar_chart(
        display_df,
        x="Model",
        y=chart_col,
        horizontal=True,
    )

    best = display_df.iloc[0]

    st.success(
        f"Highest {metric_label.lower()} in the saved results: "
        f"{best['Model']} ({best[chart_col]:.2f}%)."
    )

    st.subheader("Full metric comparison")

    full = saved_active[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Malignant Recall",
        ]
    ].copy()

    score_columns = [
        column
        for column in full.columns
        if column != "Model"
    ]

    for column in score_columns:
        full[column] = convert_to_percentage(
            full[column]
        )

    full[score_columns] = full[score_columns].round(2)

    st.dataframe(
        full,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "The table uses saved results generated by main.py. "
        "Standard precision, recall and F1-score use "
        "Benign (class 1) as the positive class."
    )


# --------------------------------------------------
# PREDICTION LAB
# --------------------------------------------------

elif page == "Prediction Lab":

    st.title("🧪 Prediction Lab")

    st.caption(
        "Choose a held-out test sample and compare predictions "
        "from the nine trained models."
    )

    st.info(
        "These predictions use samples from the built-in "
        "dataset for educational comparison only."
    )

    sample_index = st.selectbox(
        "Select a test sample",
        range(len(X_test)),
        format_func=lambda i: f"Test sample {i + 1}",
    )

    sample = X_test.iloc[
        [sample_index]
    ][active_columns]

    actual = int(
        y_test.iloc[sample_index]
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Test sample",
        f"#{sample_index + 1}",
    )

    c2.metric(
        "Actual dataset label",
        "Benign" if actual == 1 else "Malignant",
    )

    with st.expander(
        "View feature values used by the selected feature set"
    ):
        st.dataframe(
            sample.T.rename(
                columns={
                    sample.index[0]: "Value",
                }
            ),
            width="stretch",
        )

    rows = []
    probabilities = []

    for name, result in active_models.items():

        prediction = int(
            result["model"].predict(sample)[0]
        )

        proba = result["model"].predict_proba(sample)[0]

        rows.append({
            "Model": name,

            "Predicted class": (
                "Benign"
                if prediction == 1
                else "Malignant"
            ),

            "Benign probability (%)": round(
                proba[1] * 100,
                2,
            ),

            "Malignant probability (%)": round(
                proba[0] * 100,
                2,
            ),

            "Matches dataset label": (
                "Yes"
                if prediction == actual
                else "No"
            ),
        })

        probabilities.append({
            "Model": name,
            "Benign": proba[1] * 100,
            "Malignant": proba[0] * 100,
        })

    st.subheader("Predictions from all nine models")

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Predicted probability comparison")

    probability_df = pd.DataFrame(probabilities)

    st.bar_chart(
        probability_df,
        x="Model",
        y=["Benign", "Malignant"],
    )

    st.caption(
        "Displayed probabilities are model outputs, not "
        "verified clinical probabilities. This project "
        "must not be used for medical decision-making."
    )


# --------------------------------------------------
# PERFORMANCE ANALYSIS
# --------------------------------------------------

elif page == "Performance Analysis":

    st.title("📈 Performance Analysis")

    st.caption(
        f"Confusion matrices and ROC curves for "
        f"{feature_set.lower()}."
    )

    st.subheader("Confusion matrices")

    st.write(
        "Rows represent actual labels and columns represent "
        "predicted labels. Class 0 is Malignant and class 1 is Benign."
    )

    cols = st.columns(2)

    for i, (name, result) in enumerate(
        active_models.items()
    ):

        with cols[i % 2]:

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=result["confusion_matrix"],
                display_labels=[
                    "Malignant",
                    "Benign",
                ],
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
            )

            ax.set_title(name)

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

    st.subheader("ROC curve comparison")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for name, result in active_models.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            result["probabilities"],
        )

        auc = roc_auc_score(
            y_test,
            result["probabilities"],
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {auc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random baseline",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # Malignant recall from saved CSV results

    st.subheader("Malignant Recall — Saved Results")

    malignant = saved_active[
        ["Model", "Malignant Recall"]
    ].copy()

    malignant["Malignant Recall (%)"] = (
        convert_to_percentage(
            malignant.pop("Malignant Recall")
        )
    )

    malignant = malignant.dropna(
        subset=["Malignant Recall (%)"]
    ).sort_values(
        "Malignant Recall (%)",
        ascending=True,
    )

    if malignant.empty:

        st.warning(
            "No valid Malignant Recall values were found "
            "in model_comparison.csv."
        )

    else:

        st.dataframe(
            malignant.round(2),
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            malignant,
            x="Model",
            y="Malignant Recall (%)",
            horizontal=True,
        )

        best_malignant = malignant.loc[
            malignant["Malignant Recall (%)"].idxmax()
        ]

        st.success(
            f"Highest saved malignant recall: "
            f"{best_malignant['Model']} "
            f"({best_malignant['Malignant Recall (%)']:.2f}%)."
        )

    st.caption(
        "Malignant recall is the proportion of actual malignant "
        "samples correctly identified as malignant in the test "
        "split. Results are experimental, not clinical evidence."
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature Importance":

    st.title("🔍 Feature Importance")

    st.caption(
        f"Explore feature importance for tree-based models "
        f"using {feature_set.lower()}."
    )

    model_name = st.selectbox(
        "Choose a model",
        [
            "Random Forest",
            "AdaBoost",
            "Gradient Boosting",
        ],
    )

    model = active_models[model_name]["model"]

    estimator = (
        model.named_steps["model"]
        if isinstance(model, Pipeline)
        else model
    )

    if not hasattr(
        estimator,
        "feature_importances_",
    ):

        st.info(
            "Feature importance is not available for this model."
        )

    else:

        importance_df = pd.DataFrame({
            "Feature": active_columns,
            "Importance": estimator.feature_importances_,
        }).sort_values(
            "Importance",
            ascending=False,
        )

        top_n = st.slider(
            "Number of features to display",
            5,
            len(active_columns),
            min(15, len(active_columns)),
        )

        top = importance_df.head(
            top_n
        ).sort_values(
            "Importance"
        )

        fig, ax = plt.subplots(
            figsize=(10, max(5, top_n * 0.32))
        )

        ax.barh(
            top["Feature"],
            top["Importance"],
        )

        ax.set_xlabel("Feature importance")

        ax.set_title(
            f"Top {top_n} features — {model_name}"
        )

        ax.grid(axis="x", alpha=0.2)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        with st.expander(
            "View complete feature-importance table"
        ):

            st.dataframe(
                importance_df,
                hide_index=True,
                width="stretch",
            )

    st.caption(
        "Feature importance describes how a model uses "
        "input features. It does not establish causation "
        "or clinical significance."
    )


st.divider()

st.caption(
    "Breast Cancer Classification · Machine Learning Mini Project "
    "· Academic Year 2026–27"
)