from pathlib import Path
import pandas as pd


# ============================================================
# RETRAINING POLICY CONFIGURATION
# ============================================================

F1_THRESHOLD = 0.60
PRECISION_THRESHOLD = 0.55
RECALL_THRESHOLD = 0.55

ARTIFACT_DIR = Path("ml/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_value(df, possible_names):
    """
    Find a metric value from either:
    1. Wide format:
       f1, precision, recall
    2. Long format:
       metric | value
    """

    # Normalize column names
    df.columns = [
        str(col).strip().lower().replace("-", "_").replace(" ", "_")
        for col in df.columns
    ]

    # --------------------------------------------------------
    # Case 1: metric exists directly as a column
    # --------------------------------------------------------

    for name in possible_names:
        if name in df.columns:
            return float(df[name].iloc[0])

    # --------------------------------------------------------
    # Case 2: CSV uses metric/value format
    # --------------------------------------------------------

    metric_columns = [
        col for col in df.columns
        if col in ["metric", "metrics", "name"]
    ]

    value_columns = [
        col for col in df.columns
        if col in ["value", "score", "metric_value"]
    ]

    if metric_columns and value_columns:

        metric_col = metric_columns[0]
        value_col = value_columns[0]

        for name in possible_names:

            matches = df[
                df[metric_col]
                .astype(str)
                .str.lower()
                .str.replace("-", "_")
                .str.replace(" ", "_")
                == name
            ]

            if not matches.empty:
                return float(matches[value_col].iloc[0])

    return None


# ============================================================
# RETRAINING CHECK
# ============================================================

def check_retraining_conditions():

    print("\n" + "=" * 60)
    print("MODEL RETRAINING POLICY CHECK")
    print("=" * 60)

    reasons = []

    # ========================================================
    # 1. MODEL PERFORMANCE
    # ========================================================

    performance_file = ARTIFACT_DIR / "monitoring_performance.csv"

    if performance_file.exists():

        performance = pd.read_csv(performance_file)

        f1 = find_value(
            performance,
            ["f1", "f1_score", "f1score"]
        )

        precision = find_value(
            performance,
            ["precision", "precision_score"]
        )

        recall = find_value(
            performance,
            ["recall", "recall_score"]
        )

        print("\nMODEL PERFORMANCE")

        if f1 is not None:
            print(f"F1 Score   : {f1:.4f}")

            if f1 < F1_THRESHOLD:
                reasons.append(
                    f"F1 score below threshold "
                    f"({f1:.4f} < {F1_THRESHOLD})"
                )
        else:
            print("F1 Score   : Not found")

        if precision is not None:
            print(f"Precision  : {precision:.4f}")

            if precision < PRECISION_THRESHOLD:
                reasons.append(
                    f"Precision below threshold "
                    f"({precision:.4f} < {PRECISION_THRESHOLD})"
                )
        else:
            print("Precision  : Not found")

        if recall is not None:
            print(f"Recall     : {recall:.4f}")

            if recall < RECALL_THRESHOLD:
                reasons.append(
                    f"Recall below threshold "
                    f"({recall:.4f} < {RECALL_THRESHOLD})"
                )
        else:
            print("Recall     : Not found")

    else:

        print("\nPerformance report not found.")
        reasons.append("Missing model performance report")

    # ========================================================
    # 2. FEATURE DRIFT
    # ========================================================

    drift_file = ARTIFACT_DIR / "feature_drift_report.csv"

    if drift_file.exists():

        drift = pd.read_csv(drift_file)

        drift.columns = [
            str(col).strip().lower()
            .replace("-", "_")
            .replace(" ", "_")
            for col in drift.columns
        ]

        # Find drift flag column
        drift_flag_column = None

        for column in ["drift", "drift_flag", "is_drift", "drift_detected"]:
            if column in drift.columns:
                drift_flag_column = column
                break

        if drift_flag_column:

            drift_values = (
                drift[drift_flag_column]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            drifted_features = drift[drift_values]

            print("\nFEATURE DRIFT")
            print(f"Drifted features: {len(drifted_features)}")

            if not drifted_features.empty:

                if "feature" in drift.columns:

                    for feature in drifted_features["feature"]:
                        reasons.append(
                            f"Feature drift detected: {feature}"
                        )

                else:
                    reasons.append(
                        "Feature drift detected"
                    )

        else:

            print("\nFEATURE DRIFT")
            print("Drift flag column not found.")

    else:

        print("\nDrift report not found.")
        reasons.append("Missing feature drift report")

    # ========================================================
    # 3. DATA QUALITY
    # ========================================================

    quality_file = ARTIFACT_DIR / "data_quality_report.csv"

    if quality_file.exists():

        quality = pd.read_csv(quality_file)

        quality.columns = [
            str(col).strip().lower()
            .replace("-", "_")
            .replace(" ", "_")
            for col in quality.columns
        ]

        failed_quality = False

        # Check common possible column names
        for column in [
            "missing_count",
            "missing",
            "missing_values"
        ]:

            if column in quality.columns:

                if (pd.to_numeric(
                    quality[column],
                    errors="coerce"
                ).fillna(0) > 0).any():

                    failed_quality = True

        for column in [
            "infinite_count",
            "infinite",
            "infinite_values"
        ]:

            if column in quality.columns:

                if (pd.to_numeric(
                    quality[column],
                    errors="coerce"
                ).fillna(0) > 0).any():

                    failed_quality = True

        for column in [
            "duplicate_count",
            "duplicates",
            "duplicate_values"
        ]:

            if column in quality.columns:

                if (pd.to_numeric(
                    quality[column],
                    errors="coerce"
                ).fillna(0) > 0).any():

                    failed_quality = True

        print("\nDATA QUALITY")

        if failed_quality:

            print("Status: FAILED")
            reasons.append("Data quality validation failed")

        else:

            print("Status: OK")

    else:

        print("\nData quality report not found.")
        reasons.append("Missing data quality report")

    # ========================================================
    # 4. FINAL DECISION
    # ========================================================

    print("\n" + "-" * 60)

    if reasons:

        decision = "RETRAIN REQUIRED"

        print("DECISION: RETRAIN REQUIRED")

        print("\nReasons:")

        for reason in reasons:
            print(f"- {reason}")

    else:

        decision = "NO RETRAINING REQUIRED"

        print("DECISION: NO RETRAINING REQUIRED")
        print("Current model can continue serving predictions.")

    # ========================================================
    # 5. SAVE DECISION
    # ========================================================

    result = pd.DataFrame(
        {
            "decision": [decision],
            "reason_count": [len(reasons)],
            "reasons": [
                "; ".join(reasons)
                if reasons
                else "None"
            ],
        }
    )

    output_file = ARTIFACT_DIR / "retraining_decision.csv"

    result.to_csv(
        output_file,
        index=False
    )

    print(f"\nDecision saved to:")
    print(output_file)

    print("=" * 60)

    return decision


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    check_retraining_conditions()