"""Entry point for the data ingestion pipeline: bronze -> silver -> gold.

    python main.py

Everything the pipeline needs is configured in the block below.
"""

import glob
import os

import pyspark

from data_processing import bronze_table, silver_table, gold_table
from utils.utils import generate_first_of_month_dates, read_all_partitions

# set up config
START_DATE_STR = "2023-01-01"
END_DATE_STR = "2025-12-01"

DATA_DIRECTORY = "data"
DATAMART_DIRECTORY = "datamart"

# label definition: 30 days past due at 6 months on book
LABEL_DPD = 30
LABEL_MOB = 6


def summarise(directory, spark):
    """Print the row count across every partition of a gold table."""
    partitions = glob.glob(os.path.join(directory, '*'))
    if not partitions:
        print(directory, 'has no partitions')
        return

    print('===', directory, 'partitions:', len(partitions), '===')
    read_all_partitions(directory, spark).show(5)


def main():
    spark = (pyspark.sql.SparkSession.builder
             .appName("data_ingestion_pipeline")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    bronze_root = os.path.join(DATAMART_DIRECTORY, "bronze")
    silver_root = os.path.join(DATAMART_DIRECTORY, "silver")
    gold_root = os.path.join(DATAMART_DIRECTORY, "gold")

    dates_str_lst = generate_first_of_month_dates(START_DATE_STR, END_DATE_STR)
    print('processing', len(dates_str_lst), 'snapshot months:',
          dates_str_lst[0], '->', dates_str_lst[-1])

    try:
        bronze_table.run_bronze_backfill(dates_str_lst, spark,
                                         data_directory=DATA_DIRECTORY,
                                         bronze_root=bronze_root)

        silver_table.run_silver_backfill(dates_str_lst, spark,
                                         bronze_root=bronze_root,
                                         silver_root=silver_root)

        gold_table.run_gold_backfill(dates_str_lst, spark,
                                     silver_root=silver_root,
                                     gold_root=gold_root,
                                     dpd=LABEL_DPD, mob=LABEL_MOB)

        summarise(os.path.join(gold_root, "label_store"), spark)
        summarise(os.path.join(gold_root, "features_store"), spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
