"""Data processing pipeline, extracted from data_processing.ipynb.

    bronze_table  - land each source csv as a monthly csv partition, as-is
    silver_table  - enforce schema and clean each source into parquet
    gold_table    - build the label store and the feature store

Partition naming and partition read/write are shared by the layers and live in
utils/utils.py. main.py orchestrates a backfill across all three layers.
"""
