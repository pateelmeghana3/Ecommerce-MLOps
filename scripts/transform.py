import pandas as pd
import os
from datetime import datetime

INPUT_FILE = "data/processed/raw_retail.csv"
OUTPUT_FILE = "data/processed/cleaned_retail.csv"


def transform_data():

    print("Starting data transformation...")

    df = pd.read_csv(INPUT_FILE)

    print("Original rows:", len(df))

    rejected = []

    # ------------------------------------------
    # 1. Cancelled invoices
    # ------------------------------------------

    cancelled = df[
        df["InvoiceNo"].astype(str).str.startswith("C")
    ].copy()

    for _, row in cancelled.iterrows():
        rejected.append({
            "source_file": INPUT_FILE,
            "rejected_date": datetime.now(),
            "reason": "Cancelled invoice",
            "record_data": str(row.to_dict())
        })

    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    # ------------------------------------------
    # 2. Invalid quantities
    # ------------------------------------------

    invalid_quantity = df[df["Quantity"] <= 0].copy()

    for _, row in invalid_quantity.iterrows():
        rejected.append({
            "source_file": INPUT_FILE,
            "rejected_date": datetime.now(),
            "reason": "Invalid quantity",
            "record_data": str(row.to_dict())
        })

    df = df[df["Quantity"] > 0]

    # ------------------------------------------
    # 3. Invalid prices
    # ------------------------------------------

    invalid_price = df[df["UnitPrice"] <= 0].copy()

    for _, row in invalid_price.iterrows():
        rejected.append({
            "source_file": INPUT_FILE,
            "rejected_date": datetime.now(),
            "reason": "Invalid unit price",
            "record_data": str(row.to_dict())
        })

    df = df[df["UnitPrice"] > 0]

    # ------------------------------------------
    # 4. Missing CustomerID
    # ------------------------------------------

    missing_customer = df[df["CustomerID"].isna()].copy()

    for _, row in missing_customer.iterrows():
        rejected.append({
            "source_file": INPUT_FILE,
            "rejected_date": datetime.now(),
            "reason": "Missing CustomerID",
            "record_data": str(row.to_dict())
        })

    df = df.dropna(subset=["CustomerID"])

    # ------------------------------------------
    # 5. Convert date
    # ------------------------------------------

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # ------------------------------------------
    # 6. Calculate revenue
    # ------------------------------------------

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # ------------------------------------------
    # 7. Remove duplicates
    # ------------------------------------------

    duplicates = df[df.duplicated(keep="first")].copy()

    for _, row in duplicates.iterrows():
        rejected.append({
            "source_file": INPUT_FILE,
            "rejected_date": datetime.now(),
            "reason": "Duplicate record",
            "record_data": str(row.to_dict())
        })

    df = df.drop_duplicates()

    # ------------------------------------------
    # Save cleaned data
    # ------------------------------------------

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------
    # Save rejected records locally
    # ------------------------------------------

    os.makedirs("data/rejected", exist_ok=True)

    rejected_df = pd.DataFrame(rejected)

    rejected_df.to_csv(
        "data/rejected/rejected_records.csv",
        index=False
    )

    print("\nTransformation completed successfully!")
    print("--------------------------------------")
    print("Clean rows:", len(df))
    print("Rejected rows:", len(rejected))
    print("Cleaned output:", OUTPUT_FILE)
    print("Rejected output: data/rejected/rejected_records.csv")


if __name__ == "__main__":
    transform_data()