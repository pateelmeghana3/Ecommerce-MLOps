
import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DB_USER = "postgres"
DB_PASSWORD = "Meghana2123"

DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecommerce_dw"

password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://"
    f"{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

INPUT_FILE = "data/processed/cleaned_retail.csv"


# =========================================================
# MAIN FUNCTION
# =========================================================

def load_data():

    print("==============================================")
    print("INCREMENTAL DATA LOADING")
    print("==============================================")

    new_customers = pd.DataFrame()
    new_products = pd.DataFrame()
    new_geography = pd.DataFrame()
    new_dates = pd.DataFrame()

    new_fact_sales_count = 0
    duplicate_fact_count = 0

    print("\nLoading cleaned data...")
    df = pd.read_csv(INPUT_FILE)

    print("CSV rows:", len(df))

    df = df.drop_duplicates().copy()
    print("Rows after duplicate removal:", len(df))

    # Standardize data types
    for column in ["CustomerID", "Quantity", "UnitPrice", "Revenue"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "CustomerID", "InvoiceDate", "Quantity",
            "UnitPrice", "Revenue"
        ]
    ).copy()

    df["CustomerID"] = df["CustomerID"].astype(int)

    # Database transaction
    with engine.begin() as conn:

        # CUSTOMER DIMENSION
        customers = df[
            ["CustomerID", "Country"]
        ].drop_duplicates(subset=["CustomerID"]).copy()

        customers.columns = ["customer_id", "country"]

        existing_customers = pd.read_sql(
            text("SELECT customer_id FROM dim_customer"), conn
        )

        new_customers = customers[
            ~customers["customer_id"].isin(
                existing_customers["customer_id"]
            )
        ].copy()

        if not new_customers.empty:
            new_customers.to_sql(
                "dim_customer", conn, if_exists="append",
                index=False, method="multi", chunksize=5000
            )

        print("New customers inserted:", len(new_customers))

        # PRODUCT DIMENSION
        products = df[
            ["StockCode", "Description"]
        ].drop_duplicates(subset=["StockCode"]).copy()

        products.columns = ["stock_code", "description"]

        existing_products = pd.read_sql(
            text("SELECT stock_code FROM dim_product"), conn
        )

        new_products = products[
            ~products["stock_code"].isin(
                existing_products["stock_code"]
            )
        ].copy()

        if not new_products.empty:
            new_products.to_sql(
                "dim_product", conn, if_exists="append",
                index=False, method="multi", chunksize=5000
            )

        print("New products inserted:", len(new_products))

        # GEOGRAPHY DIMENSION
        geography = df[["Country"]].drop_duplicates().copy()
        geography.columns = ["country"]

        existing_geography = pd.read_sql(
            text("SELECT country FROM dim_geography"), conn
        )

        new_geography = geography[
            ~geography["country"].isin(
                existing_geography["country"]
            )
        ].copy()

        if not new_geography.empty:
            new_geography.to_sql(
                "dim_geography", conn, if_exists="append",
                index=False, method="multi", chunksize=5000
            )

        print("New geographies inserted:", len(new_geography))

        # DATE DIMENSION
        dates = pd.DataFrame({
            "full_date": df["InvoiceDate"].dt.date.unique()
        })

        dates["full_date"] = pd.to_datetime(dates["full_date"])
        dates["date_key"] = dates["full_date"].dt.strftime(
            "%Y%m%d"
        ).astype(int)
        dates["day"] = dates["full_date"].dt.day
        dates["month"] = dates["full_date"].dt.month
        dates["month_name"] = dates["full_date"].dt.month_name()
        dates["quarter"] = dates["full_date"].dt.quarter
        dates["year"] = dates["full_date"].dt.year

        dates = dates[
            [
                "date_key", "full_date", "day", "month",
                "month_name", "quarter", "year"
            ]
        ]

        existing_dates = pd.read_sql(
            text("SELECT date_key FROM dim_date"), conn
        )

        new_dates = dates[
            ~dates["date_key"].isin(existing_dates["date_key"])
        ].copy()

        if not new_dates.empty:
            new_dates.to_sql(
                "dim_date", conn, if_exists="append",
                index=False, method="multi", chunksize=5000
            )

        print("New dates inserted:", len(new_dates))

        # GET DIMENSION KEYS
        customer_keys = pd.read_sql(
            text("SELECT customer_key, customer_id FROM dim_customer"),
            conn
        )

        product_keys = pd.read_sql(
            text("SELECT product_key, stock_code FROM dim_product"),
            conn
        )

        geography_keys = pd.read_sql(
            text("SELECT geography_key, country FROM dim_geography"),
            conn
        )

        # BUILD FACT SALES
        sales = df.merge(
            customer_keys,
            left_on="CustomerID", right_on="customer_id", how="inner"
        )

        sales = sales.merge(
            product_keys,
            left_on="StockCode", right_on="stock_code", how="inner"
        )

        sales = sales.merge(
            geography_keys,
            left_on="Country", right_on="country", how="inner"
        )

        sales["date_key"] = sales["InvoiceDate"].dt.strftime(
            "%Y%m%d"
        ).astype(int)

        fact_sales = sales[
            [
                "InvoiceNo", "customer_key", "product_key",
                "date_key", "geography_key", "Quantity",
                "UnitPrice", "Revenue"
            ]
        ].copy()

        fact_sales.columns = [
            "invoice_no", "customer_key", "product_key",
            "date_key", "geography_key", "quantity",
            "unit_price", "revenue"
        ]

        fact_sales = fact_sales.drop_duplicates().copy()
        print("Incoming fact rows:", len(fact_sales))

        existing_fact_count = conn.execute(
            text("SELECT COUNT(*) FROM fact_sales")
        ).scalar()

        print("Existing fact rows:", existing_fact_count)

        # REBUILD IF EXISTING TABLE HAS MORE ROWS THAN SOURCE
        if existing_fact_count > len(fact_sales):

            print("Rebuilding fact_sales from cleaned data...")

            conn.execute(text("TRUNCATE TABLE fact_sales"))

            fact_sales.to_sql(
                "fact_sales", conn, if_exists="append",
                index=False, method="multi", chunksize=5000
            )

            new_fact_sales_count = len(fact_sales)
            duplicate_fact_count = 0

        else:
            # INCREMENTAL LOAD
            conn.execute(text("""
                CREATE TEMP TABLE fact_sales_staging AS
                SELECT * FROM fact_sales WITH NO DATA
            """))

            fact_sales.to_sql(
                "fact_sales_staging", conn, if_exists="append",
                index=False, method="multi", chunksize=5000
            )

            duplicate_fact_count = conn.execute(text("""
                SELECT COUNT(*)
                FROM fact_sales_staging s
                WHERE EXISTS (
                    SELECT 1
                    FROM fact_sales f
                    WHERE f.invoice_no = s.invoice_no
                      AND f.customer_key = s.customer_key
                      AND f.product_key = s.product_key
                      AND f.date_key = s.date_key
                      AND f.geography_key = s.geography_key
                      AND f.quantity = s.quantity
                      AND f.unit_price = s.unit_price
                      AND f.revenue = s.revenue
                )
            """)).scalar()

            new_fact_sales_count = len(fact_sales) - duplicate_fact_count

            conn.execute(text("""
                INSERT INTO fact_sales (
                    invoice_no, customer_key, product_key,
                    date_key, geography_key, quantity,
                    unit_price, revenue
                )
                SELECT
                    s.invoice_no, s.customer_key, s.product_key,
                    s.date_key, s.geography_key, s.quantity,
                    s.unit_price, s.revenue
                FROM fact_sales_staging s
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM fact_sales f
                    WHERE f.invoice_no = s.invoice_no
                      AND f.customer_key = s.customer_key
                      AND f.product_key = s.product_key
                      AND f.date_key = s.date_key
                      AND f.geography_key = s.geography_key
                      AND f.quantity = s.quantity
                      AND f.unit_price = s.unit_price
                      AND f.revenue = s.revenue
                )
            """))

        # FINAL COUNTS
        tables = [
            ("Customers", "dim_customer"),
            ("Products", "dim_product"),
            ("Geographies", "dim_geography"),
            ("Dates", "dim_date"),
            ("Fact sales", "fact_sales"),
        ]

        final_counts = {}

        for label, table in tables:
            final_counts[label] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()

    print("\n==============================================")
    print("INCREMENTAL DATA LOADING COMPLETED")
    print("==============================================")

    for label, count in final_counts.items():
        print(f"{label:<15}: {count}")

    print("\nNew records inserted:")
    print("Customers       :", len(new_customers))
    print("Products        :", len(new_products))
    print("Geographies     :", len(new_geography))
    print("Dates           :", len(new_dates))
    print("Fact sales      :", new_fact_sales_count)
    print("Duplicate facts skipped:", duplicate_fact_count)

    print("\nSUCCESS")


if __name__ == "__main__":
    load_data()
