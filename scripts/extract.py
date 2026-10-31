import pandas as pd
import os
from datetime import datetime
from sqlalchemy import create_engine
from urllib.parse import quote_plus

SOURCE_FILE = "data/raw/Online Retail.xlsx"
RAW_OUTPUT = "data/processed/raw_retail.csv"

DB_USER = "postgres"
DB_PASSWORD = "Meghana@123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecommerce_dw"

password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{password}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def extract_data():

    print("Starting data extraction...")

    extraction_date = datetime.now()

    try:

        if not os.path.exists(SOURCE_FILE):
            raise FileNotFoundError(
                f"Dataset not found: {SOURCE_FILE}"
            )

        # Read source dataset
        df = pd.read_excel(SOURCE_FILE)

        # Create output directory
        os.makedirs("data/processed", exist_ok=True)

        # Save raw copy
        df.to_csv(RAW_OUTPUT, index=False)

        row_count = len(df)

        # Record successful ingestion
        log = pd.DataFrame([{
            "source_file": SOURCE_FILE,
            "extraction_date": extraction_date,
            "row_count": row_count,
            "status": "SUCCESS",
            "error_message": None
        }])

        log.to_sql(
            "ingestion_log",
            engine,
            if_exists="append",
            index=False
        )

        print("\nExtraction completed successfully!")
        print("--------------------------------")
        print("Source:", SOURCE_FILE)
        print("Extraction date:", extraction_date)
        print("Row count:", row_count)
        print("Output:", RAW_OUTPUT)

    except Exception as e:

        print("\nExtraction failed!")
        print("Error:", e)

        # Record failed ingestion
        log = pd.DataFrame([{
            "source_file": SOURCE_FILE,
            "extraction_date": extraction_date,
            "row_count": 0,
            "status": "FAILED",
            "error_message": str(e)
        }])

        log.to_sql(
            "ingestion_log",
            engine,
            if_exists="append",
            index=False
        )


if __name__ == "__main__":
    extract_data()