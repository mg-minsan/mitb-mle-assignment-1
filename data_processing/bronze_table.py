"""Bronze layer: land one monthly csv partition per source, no cleaning."""

import os
from datetime import datetime

from pyspark.sql.functions import col

from utils.utils import write_csv_partition

# source csv -> bronze partition prefix and output sub-directory
SOURCES = {
    "lms": {
        "csv": "lms_loan_daily.csv",
        "prefix": "bronze_loan_daily_",
        "subdir": "lms",
    },
    "financials": {
        "csv": "features_financials.csv",
        "prefix": "bronze_financials_",
        "subdir": "financials",
    },
    "clickstream": {
        "csv": "feature_clickstream.csv",
        "prefix": "bronze_clickstream_",
        "subdir": "clickstream",
    },
    "customers": {
        "csv": "features_attributes.csv",
        "prefix": "bronze_customers_",
        "subdir": "customers",
    },
}


def process_bronze_table(csv_file_path, partition_prefix, snapshot_date_str,
                         bronze_directory, spark):
    """Slice one snapshot month out of a source csv and write it to bronze."""
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load data - IRL ingest from back end source system
    df = (spark.read.csv(csv_file_path, header=True, inferSchema=True)
          .filter(col('snapshot_date') == snapshot_date))
    print('loaded from:', csv_file_path, snapshot_date_str, 'row count:', df.count())

    return write_csv_partition(df, bronze_directory, partition_prefix, snapshot_date_str)


def bronze_directory(bronze_root, source):
    return os.path.join(bronze_root, SOURCES[source]["subdir"])


def run_bronze_backfill(dates_str_lst, spark, data_directory="data",
                        bronze_root="datamart/bronze"):
    """Backfill every source for every snapshot date.

    A month a source does not cover yields a header-only partition, so the
    downstream layers see an empty table rather than a missing file.
    """
    for source, source_config in SOURCES.items():
        directory = bronze_directory(bronze_root, source)
        os.makedirs(directory, exist_ok=True)

        print('=== bronze', source, '->', directory, '===')
        for date_str in dates_str_lst:
            process_bronze_table(
                csv_file_path=os.path.join(data_directory, source_config["csv"]),
                partition_prefix=source_config["prefix"],
                snapshot_date_str=date_str,
                bronze_directory=directory,
                spark=spark,
            )
