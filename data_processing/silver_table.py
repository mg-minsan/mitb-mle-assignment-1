"""Silver layer: enforce schema, clean out-of-range values, write parquet."""

import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.utils import read_csv_partition, write_parquet_partition

# silver table -> the bronze partition it reads and the silver partition it writes
TABLES = {
    "loan_daily": {"bronze_prefix": "bronze_loan_daily_", "silver_prefix": "silver_loan_daily_", "bronze_subdir": "lms"},
    "financials": {"bronze_prefix": "bronze_financials_", "silver_prefix": "silver_financials_", "bronze_subdir": "financials"},
    "customers": {"bronze_prefix": "bronze_customers_", "silver_prefix": "silver_customers_", "bronze_subdir": "customers"},
    "clickstream": {"bronze_prefix": "bronze_clickstream_", "silver_prefix": "silver_clickstream_", "bronze_subdir": "clickstream"},
}


def _cast_columns(df, column_type_map):
    for column, new_type in column_type_map.items():
        # raw numbers can carry a stray trailing underscore ("40_", "52312.68_")
        if isinstance(new_type, (IntegerType, FloatType)):
            df = df.withColumn(column, F.regexp_replace(col(column), r"^(-?\d+(\.\d+)?)_$", "$1"))
        df = (df.withColumn(column, col(column).cast(new_type))
              .withColumnRenamed(column, column.lower()))

    # final safety net: ensure every column name is lower case
    return df.toDF(*[c.lower() for c in df.columns])


def process_lms_silver_table(snapshot_date_str, bronze_lms_directory,
                             silver_loan_daily_directory, spark):
    """Loan book: enforce schema, then derive month on book and days past due."""
    df = read_csv_partition(bronze_lms_directory, "bronze_loan_daily_", snapshot_date_str, spark)

    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    df = _cast_columns(df, column_type_map)

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed",
                       F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date",
                       F.when(col("installments_missed") > 0,
                              F.add_months(col("snapshot_date"),
                                           -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd",
                       F.when(col("overdue_amt") > 0.0,
                              F.datediff(col("snapshot_date"),
                                         col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    return write_parquet_partition(df, silver_loan_daily_directory,
                                   "silver_loan_daily_", snapshot_date_str)


def process_financials_silver_table(snapshot_date_str, bronze_financials_directory,
                                    silver_financials_directory, spark):
    """Financials: enforce schema and null out values outside a plausible range."""
    df = read_csv_partition(bronze_financials_directory, "bronze_financials_", snapshot_date_str, spark)

    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }
    df = _cast_columns(df, column_type_map)

    # money columns to cents ("113781.38999999998" -> 113781.39)
    for c in ["annual_income", "monthly_inhand_salary", "outstanding_debt"]:
        df = df.withColumn(c, F.round(col(c), 2))

    # clean data: values outside the plausible range become null
    df = df.withColumn("num_bank_accounts", F.when(F.col("num_bank_accounts").between(0, 11), F.col("num_bank_accounts")))
    df = df.withColumn("num_credit_card", F.when(F.col("num_credit_card").between(0, 11), F.col("num_credit_card")))
    df = df.withColumn("interest_rate", F.when(F.col("interest_rate").between(0, 34), F.col("interest_rate")))

    # num_of_loan: recover an impossible count from the loan type list where we can
    type_of_loan_arr = split(col("type_of_loan"), ",\\s*")
    df = df.withColumn("num_of_loan",
                       F.when((F.col("num_of_loan") < 0) & (F.col("type_of_loan").isNotNull()), F.size(type_of_loan_arr))
                       .when((F.col("num_of_loan") < 0) & (F.col("type_of_loan").isNull()), F.lit(None))
                       .when((F.col("num_of_loan") > 33) & (F.col("type_of_loan").isNotNull()), F.size(type_of_loan_arr))
                       .when((F.col("num_of_loan") > 33) & (F.col("type_of_loan").isNull()), F.lit(None))
                       .otherwise(F.col("num_of_loan")))

    df = df.withColumn("delay_from_due_date", F.when(F.col("delay_from_due_date").between(0, 365), F.col("delay_from_due_date")))
    df = df.withColumn("num_of_delayed_payment", F.when(F.col("num_of_delayed_payment").between(0, 28), F.col("num_of_delayed_payment")))
    df = df.withColumn("num_credit_inquiries", F.when(F.col("num_credit_inquiries").between(0, 17), F.col("num_credit_inquiries")))
    df = df.withColumn("total_emi_per_month", F.when(F.col("total_emi_per_month").between(0, 1000), F.col("total_emi_per_month")))

    # clean data: placeholder categories become null
    df = df.withColumn("credit_mix", F.when(F.col("credit_mix") == "_", F.lit(None)).otherwise(F.col("credit_mix")))
    df = df.withColumn("payment_behaviour", F.when(F.col("payment_behaviour") == "!@9#%8", F.lit(None)).otherwise(F.col("payment_behaviour")))

    # augment data: "Low_spent_Small_value_payments" -> spend_level 0, payment_value 1 (missing stays null)
    spending_level = F.regexp_extract("payment_behaviour", r"^(Low|High)_spent", 1)
    payment_value = F.regexp_extract("payment_behaviour", r"_(Small|Medium|Large)_value", 1)
    df = df.withColumn("spend_level",
                       F.when(spending_level == "Low", 0).when(spending_level == "High", 1).cast(IntegerType()))
    df = df.withColumn("payment_value",
                       F.when(payment_value == "Small", 1).when(payment_value == "Medium", 2)
                        .when(payment_value == "Large", 3).cast(IntegerType()))

    # augment data: "22 Years and 1 Months" -> 265
    df = df.withColumn(
        "credit_history_months",
        F.regexp_extract("credit_history_age", r"(\d+)\s*Years?", 1).cast("int") * 12
        + F.regexp_extract("credit_history_age", r"(\d+)\s*Months?", 1).cast("int")
    )

    return write_parquet_partition(df, silver_financials_directory,
                                   "silver_financials_", snapshot_date_str)


def process_customers_silver_table(snapshot_date_str, bronze_customers_directory,
                                   silver_customers_directory, spark):
    """Customer attributes: enforce schema and null out malformed values."""
    df = read_csv_partition(bronze_customers_directory, "bronze_customers_", snapshot_date_str, spark)

    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    df = _cast_columns(df, column_type_map)

    # under-18 ages are invalid: no minors can borrow, and they carry ~12 years of credit history
    df = df.withColumn("age", F.when(F.col("age").between(18, 100), F.col("age")))
    df = df.withColumn("ssn", F.when(F.col("ssn").rlike(r"^\d{3}-\d{2}-\d{4}$"), F.col("ssn")))
    df = df.withColumn("occupation", F.when(F.col("occupation").rlike("^_+$"), None).otherwise(F.col("occupation")))

    return write_parquet_partition(df, silver_customers_directory,
                                   "silver_customers_", snapshot_date_str)


def process_clickstream_silver_table(snapshot_date_str, bronze_clickstream_directory,
                                     silver_clickstream_directory, spark):
    """Clickstream: schema enforcement only, the features arrive clean."""
    df = read_csv_partition(bronze_clickstream_directory, "bronze_clickstream_", snapshot_date_str, spark)

    column_type_map = {f"fe_{i}": IntegerType() for i in range(1, 21)}
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()
    df = _cast_columns(df, column_type_map)

    return write_parquet_partition(df, silver_clickstream_directory,
                                   "silver_clickstream_", snapshot_date_str)


PROCESSORS = {
    "loan_daily": process_lms_silver_table,
    "financials": process_financials_silver_table,
    "customers": process_customers_silver_table,
    "clickstream": process_clickstream_silver_table,
}


def silver_directory(silver_root, table):
    return os.path.join(silver_root, table)


def run_silver_backfill(dates_str_lst, spark, bronze_root="datamart/bronze",
                        silver_root="datamart/silver"):
    """Backfill every silver table for every snapshot date."""
    for table, table_config in TABLES.items():
        directory = silver_directory(silver_root, table)
        os.makedirs(directory, exist_ok=True)

        print('=== silver', table, '->', directory, '===')
        for date_str in dates_str_lst:
            PROCESSORS[table](
                date_str,
                os.path.join(bronze_root, table_config["bronze_subdir"]),
                directory,
                spark,
            )
