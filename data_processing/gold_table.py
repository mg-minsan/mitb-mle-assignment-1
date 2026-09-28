"""Gold layer: the label store and the feature store."""

import os
from datetime import datetime
from functools import reduce

import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

from utils.utils import (generate_first_of_month_dates, partition_path,
                         read_parquet_partition, write_parquet_partition)

FE_COLUMNS = [f"fe_{i}" for i in range(1, 21)]


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory,
                              gold_label_store_directory, spark, dpd, mob):
    """Label a loan as default when it is dpd days past due at month mob."""
    df = read_parquet_partition(silver_loan_daily_directory, "silver_loan_daily_",
                                snapshot_date_str, spark)

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd) + 'dpd_' + str(mob) + 'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    return write_parquet_partition(df, gold_label_store_directory,
                                   "gold_label_store_", snapshot_date_str)


def aggregate_clickstream_features(snapshot_date_str, silver_clickstream_directory, spark, months=6):
    """Mean of each clickstream feature over the loan-start month and the months before it."""
    # window: loan-start month and the (months - 1) before it, never after
    start_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d") - relativedelta(months=months - 1)
    window_dates = generate_first_of_month_dates(start_date.strftime("%Y-%m-%d"), snapshot_date_str)

    # early loans have a short window: clickstream starts 2023-01
    window_dates = [d for d in window_dates
                    if os.path.exists(partition_path(silver_clickstream_directory, "silver_clickstream_", d, ".parquet"))]
    if not window_dates:
        return None

    stacked = reduce(lambda a, b: a.unionByName(b),
                     [read_parquet_partition(silver_clickstream_directory, "silver_clickstream_", d, spark)
                      for d in window_dates])

    # leakage guard: nothing after the loan starts
    assert stacked.filter(col("snapshot_date") > F.lit(snapshot_date_str).cast("date")).count() == 0

    return (stacked.groupBy("customer_id")
            .agg(*[F.avg(c).alias(f"{c}_mean_{months}m") for c in FE_COLUMNS],
                 F.count("*").cast(IntegerType()).alias(f"cs_months_{months}m")))


def process_features_gold_table(snapshot_date_str, silver_customers_directory,
                                silver_financials_directory, silver_clickstream_directory,
                                gold_features_store_directory, spark):
    """One row per applicant in their application month, no loan columns so nothing can leak."""
    # customers is the spine: one row per applicant, dated at loan start
    df = read_parquet_partition(silver_customers_directory, "silver_customers_",
                                snapshot_date_str, spark).drop("name", "ssn")
    financials_df = read_parquet_partition(silver_financials_directory, "silver_financials_",
                                           snapshot_date_str, spark).drop("credit_history_age", "payment_behaviour", "type_of_loan")
    clickstream_df = aggregate_clickstream_features(snapshot_date_str, silver_clickstream_directory, spark)

    # same month join: every feature is known at application time
    df = df.join(financials_df, on=["customer_id", "snapshot_date"], how="left")

    if clickstream_df is not None:
        # clickstream is already one row per customer, so join on customer only
        df = df.join(clickstream_df, on="customer_id", how="left")
        df = df.fillna(0, subset=["cs_months_6m"])
    else:
        df = df.withColumn("cs_months_6m", F.lit(0).cast(IntegerType()))

    df = df.withColumn("has_clickstream", (col("cs_months_6m") > 0).cast(IntegerType()))
    df = df.drop("cs_months_6m")

    df = write_parquet_partition(df, gold_features_store_directory,
                                 "gold_features_store_", snapshot_date_str)

    return df


def run_gold_backfill(dates_str_lst, spark, silver_root="datamart/silver",
                      gold_root="datamart/gold", dpd=30, mob=6):
    silver_loan_daily_directory = os.path.join(silver_root, "loan_daily")
    silver_customers_directory = os.path.join(silver_root, "customers")
    silver_financials_directory = os.path.join(silver_root, "financials")
    silver_clickstream_directory = os.path.join(silver_root, "clickstream")

    gold_label_store_directory = os.path.join(gold_root, "label_store")
    gold_features_store_directory = os.path.join(gold_root, "features_store")
    os.makedirs(gold_label_store_directory, exist_ok=True)
    os.makedirs(gold_features_store_directory, exist_ok=True)

    print('=== gold label_store ->', gold_label_store_directory, '===')
    for date_str in dates_str_lst:
        process_labels_gold_table(date_str, silver_loan_daily_directory,
                                  gold_label_store_directory, spark, dpd=dpd, mob=mob)

    print('=== gold features_store ->', gold_features_store_directory, '===')
    for date_str in dates_str_lst:
        process_features_gold_table(date_str, silver_customers_directory, silver_financials_directory,
                                    silver_clickstream_directory,
                                    gold_features_store_directory, spark)
