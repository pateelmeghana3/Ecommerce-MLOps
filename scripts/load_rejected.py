import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

DB_USER = "postgres"
DB_PASSWORD = "Meghana@123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecommerce_dw"

REJECTED_FILE = "data/rejected/rejected_records.csv"

password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{password}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Loading rejected records...")

df = pd.read_csv(REJECTED_FILE)

print("Rejected records found:", len(df))

df.to_sql(
    "rejected_records",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=5000
)

print("Rejected records loaded successfully!")