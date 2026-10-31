import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "data",
    "customer_churn_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "best_churn_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "features.pkl"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "artifacts"
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

DRIFT_THRESHOLD = 0.20
PERFORMANCE_THRESHOLD = 0.60

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "purchase_days",
    "customer_lifetime_days"
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading monitoring dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset rows: {len(df)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = joblib.load(
    MODEL_PATH
)

saved_features = joblib.load(
    FEATURES_PATH
)

print(
    f"Model loaded successfully."
)

print(
    f"Features: {saved_features}"
)


# ============================================================
# DATA QUALITY MONITORING
# ============================================================

print("\n")
print("=" * 50)
print("DATA QUALITY MONITORING")
print("=" * 50)

quality_results = []


for feature in FEATURES:

    missing_count = int(
        df[feature].isna().sum()
    )

    infinite_count = int(
        np.isinf(
            df[feature]
        ).sum()
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    quality_results.append({
        "Feature": feature,
        "Missing_Values": missing_count,
        "Infinite_Values": infinite_count,
        "Duplicate_Rows": duplicate_count
    })

    print(
        f"{feature}: "
        f"missing={missing_count}, "
        f"infinite={infinite_count}"
    )


quality_df = pd.DataFrame(
    quality_results
)

quality_path = os.path.join(
    ARTIFACT_DIR,
    "data_quality_report.csv"
)

quality_df.to_csv(
    quality_path,
    index=False
)


# ============================================================
# FEATURE DRIFT MONITORING
# ============================================================

print("\n")
print("=" * 50)
print("FEATURE DRIFT MONITORING")
print("=" * 50)

drift_results = []


for feature in FEATURES:

    values = df[feature].dropna()

    if len(values) == 0:

        drift_score = 1.0

    else:

        # Compare early and recent portions
        split_index = int(
            len(values) * 0.50
        )

        baseline = values.iloc[
            :split_index
        ]

        current = values.iloc[
            split_index:
        ]

        baseline_mean = baseline.mean()
        current_mean = current.mean()

        baseline_std = baseline.std()

        if baseline_std == 0:

            drift_score = 0.0

        else:

            drift_score = abs(
                current_mean - baseline_mean
            ) / baseline_std


    drift_detected = (
        drift_score > DRIFT_THRESHOLD
    )

    drift_results.append({
        "Feature": feature,
        "Drift_Score": drift_score,
        "Drift_Threshold": DRIFT_THRESHOLD,
        "Drift_Detected": drift_detected
    })

    print(
        f"{feature}: "
        f"drift_score={drift_score:.4f}, "
        f"drift={drift_detected}"
    )


drift_df = pd.DataFrame(
    drift_results
)

drift_path = os.path.join(
    ARTIFACT_DIR,
    "feature_drift_report.csv"
)

drift_df.to_csv(
    drift_path,
    index=False
)


# ============================================================
# MODEL PERFORMANCE MONITORING
# ============================================================

print("\n")
print("=" * 50)
print("MODEL PERFORMANCE MONITORING")
print("=" * 50)

X = df[FEATURES]
y = df["churn"]


# Generate predictions

predictions = model.predict(
    X
)

probabilities = model.predict_proba(
    X
)[:, 1]


# Calculate metrics

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    probabilities
)


performance_df = pd.DataFrame([
    {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }
])


performance_path = os.path.join(
    ARTIFACT_DIR,
    "monitoring_performance.csv"
)

performance_df.to_csv(
    performance_path,
    index=False
)


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
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# MONITORING STATUS
# ============================================================

drift_count = int(
    drift_df["Drift_Detected"].sum()
)

performance_warning = (
    f1 < PERFORMANCE_THRESHOLD
)

if drift_count > 0:

    drift_status = "WARNING"

else:

    drift_status = "OK"


if performance_warning:

    performance_status = "WARNING"

else:

    performance_status = "OK"


# ============================================================
# FINAL MONITORING REPORT
# ============================================================

report_path = os.path.join(
    ARTIFACT_DIR,
    "monitoring_report.txt"
)

with open(
    report_path,
    "w"
) as report:

    report.write(
        "CUSTOMER CHURN MODEL MONITORING REPORT\n"
    )

    report.write(
        "========================================\n\n"
    )

    report.write(
        f"Dataset rows: {len(df)}\n\n"
    )

    report.write(
        "DATA QUALITY\n"
    )

    report.write(
        "------------\n"
    )

    report.write(
        f"Missing values checked: YES\n"
    )

    report.write(
        f"Infinite values checked: YES\n"
    )

    report.write(
        f"Duplicate rows checked: YES\n\n"
    )

    report.write(
        "FEATURE DRIFT\n"
    )

    report.write(
        "-------------\n"
    )

    report.write(
        f"Drift threshold: {DRIFT_THRESHOLD}\n"
    )

    report.write(
        f"Features with drift: {drift_count}\n"
    )

    report.write(
        f"Drift status: {drift_status}\n\n"
    )

    report.write(
        "MODEL PERFORMANCE\n"
    )

    report.write(
        "-----------------\n"
    )

    report.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    report.write(
        f"Precision: {precision:.4f}\n"
    )

    report.write(
        f"Recall: {recall:.4f}\n"
    )

    report.write(
        f"F1: {f1:.4f}\n"
    )

    report.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    report.write(
        f"Performance threshold: "
        f"{PERFORMANCE_THRESHOLD}\n"
    )

    report.write(
        f"Performance status: "
        f"{performance_status}\n\n"
    )

    report.write(
        "RETRAINING RECOMMENDATION\n"
    )

    report.write(
        "--------------------------\n"
    )

    if (
        drift_count > 0
        or performance_warning
    ):

        report.write(
            "RETRAINING RECOMMENDED\n"
        )

        report.write(
            "Reason: feature drift or "
            "performance degradation detected.\n"
        )

    else:

        report.write(
            "NO RETRAINING REQUIRED\n"
        )

        report.write(
            "Model monitoring is within "
            "defined thresholds.\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 50)
print("MONITORING COMPLETED")
print("=" * 50)

print(
    f"Drift status       : {drift_status}"
)

print(
    f"Performance status : {performance_status}"
)

print(
    f"F1 score           : {f1:.4f}"
)

print(
    f"Features with drift: {drift_count}"
)

print("\nReports created:")

print(
    quality_path
)

print(
    drift_path
)

print(
    performance_path
)

print(
    report_path
)

print("\nMonitoring completed successfully.")