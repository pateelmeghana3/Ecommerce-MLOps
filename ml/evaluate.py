import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "ml/data/customer_churn_dataset.csv"
MODEL_PATH = "ml/models/best_churn_model.pkl"
ARTIFACT_DIR = "ml/artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Customers loaded: {len(df)}"
)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "purchase_days",
    "customer_lifetime_days",
]

X = df[FEATURES]
y = df["churn"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Retained",
            "Churned"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# 11. SAVE CONFUSION MATRIX IMAGE
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Retained",
        "Churned"
    ]
)

disp.plot()

plt.title(
    "Customer Churn - Confusion Matrix"
)

plt.tight_layout()

confusion_path = os.path.join(
    ARTIFACT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()


# ============================================================
# 12. SAVE METRICS CSV
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_path = os.path.join(
    ARTIFACT_DIR,
    "evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 13. SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Retained",
        "Churned"
    ],
    zero_division=0
)

report_path = os.path.join(
    ARTIFACT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(report)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print("\nSaved files:")

print(
    confusion_path
)

print(
    metrics_path
)

print(
    report_path
)