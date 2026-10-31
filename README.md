# E-Commerce Sales Analytics and Customer Churn Prediction

## Project Overview

This project is an end-to-end Data Engineering and MLOps application designed to analyze e-commerce sales data, monitor business performance, and predict customer churn.

It combines data ingestion, data cleaning, ETL pipeline automation, data warehousing, interactive dashboards, REST APIs, and machine learning workflows to transform raw retail data into useful business insights.

## Objectives

- Analyze e-commerce sales and revenue trends.
- Build an automated ETL pipeline for data processing.
- Store cleaned and transformed data in PostgreSQL.
- Create interactive dashboards for business analytics.
- Develop machine learning models for customer churn prediction.
- Track and manage ML experiments using MLflow.
- Automate workflows using Apache Airflow.
- Expose application functionality through FastAPI.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas, NumPy | Data cleaning and processing |
| PostgreSQL | Data warehouse |
| Apache Airflow | ETL pipeline orchestration |
| Streamlit | Interactive analytics dashboard |
| FastAPI | REST API development |
| Scikit-learn | Machine learning |
| MLflow | Experiment tracking and model management |
| Docker | Containerization |
| Git & GitHub | Version control |

## Dataset

**Dataset:** UCI Online Retail Dataset

The dataset contains e-commerce transaction records with the following columns:

- InvoiceNo – Invoice number
- StockCode – Product code
- Description – Product description
- Quantity – Number of products purchased
- InvoiceDate – Transaction date
- UnitPrice – Price per product
- CustomerID – Customer identification
- Country – Customer location

## Project Workflow

### 1. Data Ingestion
- Collected the Online Retail dataset in Excel format.
- Loaded raw data using Python and Pandas.
- Stored the original dataset in the raw data directory.
- Maintained ingestion logs to track pipeline execution.

### 2. Data Cleaning and Transformation
- Handled missing values and invalid records.
- Removed duplicate transactions.
- Filtered invalid quantities and prices.
- Calculated revenue using quantity and unit price.
- Separated valid and rejected records.

### 3. Data Warehousing
Used PostgreSQL to create a structured data warehouse with a star-schema design.

**Dimension Tables:**
- dim_customer
- dim_product
- dim_date
- dim_geography

**Fact Table:**
- fact_sales

Additional analytical tables:
- customer_mart
- sales_mart
- ingestion_log
- rejected_records

### 4. ETL Pipeline Automation – Apache Airflow
- Created DAGs to automate the data pipeline.
- Organized ingestion, transformation, and loading tasks.
- Monitored pipeline execution through the Airflow UI.
- Used task dependencies to maintain the execution order.

### 5. Interactive Dashboard – Streamlit
Developed an interactive dashboard to visualize:

- Total revenue and sales performance.
- Number of orders and customers.
- Average order value.
- Product-wise sales analysis.
- Country-wise revenue.
- Customer purchasing behaviour.
- Sales trends over time.

### 6. REST API – FastAPI
- Developed API endpoints to expose application and analytical functionality.
- Used FastAPI for request handling and data communication.
- Enabled integration between backend services and other application components.

### 7. Machine Learning and MLOps
- Designed the customer churn prediction component.
- Used Scikit-learn for machine learning model development.
- Used MLflow for experiment tracking and model management.
- Evaluated model performance using classification metrics.
- Organized the model workflow for reproducibility.

### 8. Containerization – Docker
- Used Docker to package application components.
- Created a consistent environment for running project services.
- Supported easier setup and deployment of the application.

## Project Architecture

Raw Excel Dataset  
↓  
Python / Pandas – Data Ingestion  
↓  
Data Cleaning and Transformation  
↓  
PostgreSQL Data Warehouse  
↓  
Apache Airflow – Pipeline Orchestration  
↓  
Data Marts  
↓  
Streamlit Dashboard / FastAPI  
↓  
Machine Learning and MLflow – Churn Prediction and Experiment Tracking  
↓  
Docker – Containerization

## Project Structure

```text
Ecommerce-Sales-Analytics/
│
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx
│   └── rejected/
│       └── rejected_records.csv
│
├── dags/
│   └── ecommerce_pipeline.py
│
├── src/
│   ├── extract.py
│   ├── transform.py
│   └── load.py
│
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
```

*Note: Adjust the folder and file names above to match your actual repository.*

## Tools and Technologies

- Python for data engineering and ML development.
- PostgreSQL for structured data storage.
- Apache Airflow for workflow automation.
- Streamlit for data visualization.
- FastAPI for REST API services.
- MLflow for ML experiment tracking.
- Docker for containerization.

## Expected Outcomes

- Automated and organized data processing.
- Structured sales data warehouse.
- Interactive business intelligence dashboard.
- Improved visibility into customer purchasing patterns.
- Machine learning-based customer churn analysis.
- Reproducible and manageable ML workflows.

## Future Enhancements

- Improve churn prediction using additional customer behaviour features.
- Integrate real-time sales data.
- Deploy the application on a cloud platform.
- Add automated model retraining and monitoring.
- Improve dashboard analytics with additional business KPIs.

## Author

**Meghana Reddy**  
CSE – Artificial Intelligence and Machine Learning  
CMR Institute of Technology, Bengaluru

GitHub: https://github.com/pateelmeghana3