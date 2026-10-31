import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

DB_USER = "postgres"
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecommerce_dw"

if not DB_PASSWORD:
    raise ValueError("POSTGRES_PASSWORD environment variable is not set.")

encoded_password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{encoded_password}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Loading sales data from fact_sales...")

query = """
SELECT
    customer_key,
    date_key,
    quantity,
    unit_price,
    revenue
FROM fact_sales
WHERE customer_key IS NOT NULL
"""

df = pd.read_sql(query, engine)

print(f"Sales records loaded: {len(df)}")


# ------------------------------------------
# Load customer and date information
# ------------------------------------------

customer_query = """
SELECT
    customer_key,
    customer_id
FROM dim_customer
"""

date_query = """
SELECT
    date_key,
    full_date
FROM dim_date
"""

customers = pd.read_sql(customer_query, engine)
dates = pd.read_sql(date_query, engine)

df = df.merge(
    customers,
    on="customer_key",
    how="left"
)

df = df.merge(
    dates,
    on="date_key",
    how="left"
)

df["full_date"] = pd.to_datetime(df["full_date"])

print(f"Customers found: {df['customer_id'].nunique()}")


# ------------------------------------------
# Choose observation and prediction windows
# ------------------------------------------

max_date = df["full_date"].max()

PREDICTION_DAYS = 90
OBSERVATION_DAYS = 365

observation_end = max_date - pd.Timedelta(
    days=PREDICTION_DAYS
)

observation_start = observation_end - pd.Timedelta(
    days=OBSERVATION_DAYS
)

print("\n================================")
print("CHURN DATASET CREATION")
print("================================")

print(
    f"Observation period: "
    f"{observation_start.date()} "
    f"to {observation_end.date()}"
)

print(
    f"Prediction period: "
    f"{(observation_end + pd.Timedelta(days=1)).date()} "
    f"to {max_date.date()}"
)


# ------------------------------------------
# Observation-period transactions
# ------------------------------------------

observation_df = df[
    (df["full_date"] >= observation_start)
    & (df["full_date"] <= observation_end)
].copy()


# ------------------------------------------
# Future 90-day transactions
# ------------------------------------------

future_df = df[
    (df["full_date"] > observation_end)
    & (df["full_date"] <= max_date)
].copy()

print(
    f"\nObservation transactions: "
    f"{len(observation_df)}"
)

print(
    f"Future transactions: "
    f"{len(future_df)}"
)


# ------------------------------------------
# Create customer features
# ------------------------------------------

reference_date = observation_end

features = observation_df.groupby(
    "customer_id"
).agg(
    frequency=("date_key", "count"),
    monetary=("revenue", "sum"),
    purchase_days=("full_date", "nunique"),
    first_purchase_date=("full_date", "min"),
    last_purchase_date=("full_date", "max")
).reset_index()


# ------------------------------------------
# Calculate recency
# ------------------------------------------

features["recency"] = (
    reference_date -
    features["last_purchase_date"]
).dt.days


# ------------------------------------------
# Calculate customer lifetime
# ------------------------------------------

features["customer_lifetime_days"] = (
    features["last_purchase_date"] -
    features["first_purchase_date"]
).dt.days


# ------------------------------------------
# Create future churn label
# ------------------------------------------

future_customers = set(
    future_df["customer_id"].dropna().unique()
)

features["churn"] = (
    ~features["customer_id"].isin(future_customers)
).astype(int)


# ------------------------------------------
# Select final ML columns
# ------------------------------------------

feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "purchase_days",
    "customer_lifetime_days"
]

ml_df = features[
    [
        "customer_id",
        "last_purchase_date"
    ] + feature_columns + ["churn"]
].copy()


# ------------------------------------------
# Remove invalid values
# ------------------------------------------

ml_df = ml_df.dropna()

ml_df["last_purchase_date"] = pd.to_datetime(
    ml_df["last_purchase_date"]
)


# ------------------------------------------
# Save dataset
# ------------------------------------------

output_path = "ml/data/customer_churn_dataset.csv"

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

ml_df.to_csv(
    output_path,
    index=False
)


# ------------------------------------------
# Results
# ------------------------------------------

print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print(f"Total customers: {len(ml_df)}")

print(
    f"Churned customers: "
    f"{ml_df['churn'].sum()}"
)

print(
    f"Retained customers: "
    f"{(ml_df['churn'] == 0).sum()}"
)

print("\nChurn distribution:")
print(ml_df["churn"].value_counts())

print("\nFinal columns:")
print(ml_df.columns.tolist())

print("\nSample:")
print(ml_df.head())

print("\nDataset saved to:")
print(output_path)