import glob
import os
from datetime import datetime


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))

        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def partition_name(prefix, snapshot_date_str, extension):
    return prefix + snapshot_date_str.replace('-', '_') + extension


def partition_path(directory, prefix, snapshot_date_str, extension):
    return os.path.join(directory, partition_name(prefix, snapshot_date_str, extension))


def read_csv_partition(directory, prefix, snapshot_date_str, spark):
    filepath = partition_path(directory, prefix, snapshot_date_str, '.csv')
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    return df


def read_parquet_partition(directory, prefix, snapshot_date_str, spark):
    filepath = partition_path(directory, prefix, snapshot_date_str, '.parquet')
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    return df


def read_all_partitions(directory, spark):
    files_list = glob.glob(os.path.join(directory, '*'))
    df = spark.read.option("header", "true").parquet(*files_list)
    print('loaded from:', directory, 'row count:', df.count())
    return df


def write_csv_partition(df, directory, prefix, snapshot_date_str):
    filepath = partition_path(directory, prefix, snapshot_date_str, '.csv')
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    return df


def write_parquet_partition(df, directory, prefix, snapshot_date_str):
    filepath = partition_path(directory, prefix, snapshot_date_str, '.parquet')
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df
