import os
import joblib
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "data",
    "customer_churn_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "artifacts"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading churn dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset rows: {len(df)}")


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "purchase_days",
    "customer_lifetime_days"
]

TARGET = "churn"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Customer_Churn_Prediction"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

best_model = None
best_model_name = None
best_f1 = -1


for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training: {name}")
    print("=" * 50)

    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(
            X_train,
            y_train
        )

        # Predictions
        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )


        # ----------------------------------------------------
        # MLflow parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            name
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES)
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )


        # Model-specific parameters

        if name == "Logistic Regression":

            mlflow.log_param(
                "max_iter",
                1000
            )

        elif name == "Random Forest":

            mlflow.log_param(
                "n_estimators",
                200
            )

            mlflow.log_param(
                "class_weight",
                "balanced"
            )

        elif name == "XGBoost":

            mlflow.log_param(
                "n_estimators",
                200
            )

            mlflow.log_param(
                "max_depth",
                5
            )

            mlflow.log_param(
                "learning_rate",
                0.05
            )

            mlflow.log_param(
                "subsample",
                0.8
            )

            mlflow.log_param(
                "colsample_bytree",
                0.8
            )


        # ----------------------------------------------------
        # MLflow metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        # Save results

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        })


        # Display metrics

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


        # ----------------------------------------------------
        # Select best model
        # ----------------------------------------------------

        if f1 > best_f1:

            best_f1 = f1
            best_model = model
            best_model_name = name


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = os.path.join(
    ARTIFACT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\n")
print("=" * 40)
print("MODEL COMPARISON")
print("=" * 40)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_churn_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

features_path = os.path.join(
    MODEL_DIR,
    "features.pkl"
)

joblib.dump(
    FEATURES,
    features_path
)


# ============================================================
# SAVE MODEL METRICS
# ============================================================

metrics_path = os.path.join(
    BASE_DIR,
    "ml",
    "model_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "CUSTOMER CHURN MODEL RESULTS\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        f"Best Model: {best_model_name}\n"
    )

    f.write(
        f"Best F1 Score: {best_f1:.4f}\n"
    )


# ============================================================
# MLFLOW MODEL REGISTRY
# ============================================================

print("\n")
print("=" * 40)
print("REGISTERING BEST MODEL")
print("=" * 40)

MODEL_NAME = "Customer_Churn_Model"

with mlflow.start_run(
    run_name="Best_Model_Registration"
) as registration_run:

    run_id = registration_run.info.run_id

    # --------------------------------------------------------
    # Log metadata
    # --------------------------------------------------------

    mlflow.log_param(
        "best_model",
        best_model_name
    )

    mlflow.log_param(
        "feature_count",
        len(FEATURES)
    )

    mlflow.log_param(
        "features",
        ",".join(FEATURES)
    )

    mlflow.log_metric(
        "best_f1",
        best_f1
    )

    # --------------------------------------------------------
    # Log the joblib model as an artifact
    # --------------------------------------------------------

    mlflow.log_artifact(
        best_model_path,
        artifact_path="joblib_model"
    )

    # --------------------------------------------------------
    # Log an MLflow PyFunc model
    # --------------------------------------------------------

    class ChurnModelWrapper(mlflow.pyfunc.PythonModel):

        def load_context(self, context):

            self.model = joblib.load(
                context.artifacts["model"]
            )

        def predict(
            self,
            context,
            model_input
        ):

            return self.model.predict(
                model_input
            )

    mlflow.pyfunc.log_model(
        artifact_path="registered_model",
        python_model=ChurnModelWrapper(),
        artifacts={
            "model": best_model_path
        }
    )

    # --------------------------------------------------------
    # Get the current run
    # --------------------------------------------------------

    client = MlflowClient()

    run = client.get_run(
        run_id
    )

    # Find the logged model URI
    logged_model = None

    if hasattr(
        run,
        "outputs"
    ) and run.outputs:

        for output in run.outputs:

            if hasattr(
                output,
                "model_id"
            ):

                logged_model = output
                break

    # --------------------------------------------------------
    # Register the model
    # --------------------------------------------------------

    if logged_model:

        model_id = logged_model.model_id

        print(
            f"Logged model ID: {model_id}"
        )

        registered_model = mlflow.register_model(
            model_uri=f"models:/{model_id}",
            name=MODEL_NAME
        )

    else:

        # Fallback for MLflow installations where
        # logged model output is not exposed directly.

        model_uri = (
            f"runs:/{run_id}/registered_model"
        )

        try:

            client.get_registered_model(
                MODEL_NAME
            )

            print(
                f"Registered model '{MODEL_NAME}' already exists."
            )

        except Exception:

            client.create_registered_model(
                MODEL_NAME
            )

            print(
                f"Created registered model '{MODEL_NAME}'."
            )

        registered_model = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )


    # --------------------------------------------------------
    # Registration result
    # --------------------------------------------------------

    print("\n========================================")
    print("MODEL REGISTRATION COMPLETED")
    print("========================================")

    print(
        f"Model name    : {MODEL_NAME}"
    )

    print(
        f"Model version : {registered_model.version}"
    )

    print(
        f"Run ID        : {run_id}"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 40)
print("BEST MODEL")
print("=" * 40)

print(
    f"Model    : {best_model_name}"
)

print(
    f"F1 Score : {best_f1:.4f}"
)

print("\nModel saved to:")
print(best_model_path)

print("\nFeature list saved to:")
print(features_path)

print("\nMLflow registered model:")
print(MODEL_NAME)

print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")