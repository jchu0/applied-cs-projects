"""Main lakehouse processor implementing medallion architecture."""

import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# Lazy imports for pyspark - allows module to be imported without pyspark
PYSPARK_AVAILABLE = False
try:
    from delta import DeltaTable
    from pyspark.sql import DataFrame, SparkSession, Window
    from pyspark.sql.functions import (
        col,
        coalesce,
        current_date,
        current_timestamp,
        input_file_name,
        lit,
        max,
        row_number,
        sha2,
        to_date,
        when,
        concat_ws,
    )
    from pyspark.sql.types import StructType
    PYSPARK_AVAILABLE = True
except ImportError:
    # Create placeholder types for when pyspark is not available
    DeltaTable = None
    DataFrame = Any
    SparkSession = Any
    Window = None
    StructType = Any
    # Function placeholders
    col = coalesce = current_date = current_timestamp = None
    input_file_name = lit = max = row_number = sha2 = to_date = when = concat_ws = None

from lakehouse.config import LakehouseConfig, Layer


def _require_pyspark():
    """Raise an error if pyspark is not available."""
    if not PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for LakehouseProcessor. "
            "Install it with: pip install pyspark>=3.4.0 delta-spark>=2.4.0"
        )

# Alias for backward compatibility with tests
MedallionLayer = Layer


class LakehouseProcessor:
    """Process data through medallion architecture layers."""

    def __init__(self, spark: "SparkSession", config: Optional[LakehouseConfig] = None):
        _require_pyspark()
        self.spark = spark
        self.config = config

    def bronze_ingestion(
        self,
        source_path: str,
        bronze_path: str,
        source_name: str,
        schema: Optional[StructType] = None,
        file_format: str = "json",
    ) -> None:
        """
        Ingest raw data into bronze layer with metadata.

        Args:
            source_path: Path to source data
            bronze_path: Path to bronze table
            source_name: Name of the data source
            schema: Optional schema for the source data
            file_format: Format of source files (json, csv, parquet)
        """
        # Read raw data (schema inference or provided)
        reader = self.spark.read
        if schema:
            reader = reader.schema(schema)

        if file_format == "json":
            df = reader.json(source_path)
        elif file_format == "csv":
            df = reader.option("header", "true").csv(source_path)
        elif file_format == "parquet":
            df = reader.parquet(source_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Add bronze metadata columns
        df_bronze = (
            df.withColumn("_source", lit(source_name))
            .withColumn("_ingestion_ts", current_timestamp())
            .withColumn("_batch_id", lit(str(uuid.uuid4())))
            .withColumn("_input_file", input_file_name())
            .withColumn("_ingestion_date", to_date(current_timestamp()))
        )

        # Write to bronze (append-only)
        df_bronze.write.format("delta").mode("append").partitionBy(
            "_ingestion_date"
        ).save(bronze_path)

    def bronze_ingestion_df(
        self,
        df: DataFrame,
        bronze_path: str,
        source_name: str,
    ) -> None:
        """
        Ingest DataFrame into bronze layer with metadata.

        Args:
            df: Source DataFrame
            bronze_path: Path to bronze table
            source_name: Name of the data source
        """
        # Add bronze metadata columns
        df_bronze = (
            df.withColumn("_source", lit(source_name))
            .withColumn("_ingestion_ts", current_timestamp())
            .withColumn("_batch_id", lit(str(uuid.uuid4())))
            .withColumn("_ingestion_date", to_date(current_timestamp()))
        )

        # Write to bronze (append-only)
        df_bronze.write.format("delta").mode("append").partitionBy(
            "_ingestion_date"
        ).save(bronze_path)

    def bronze_to_silver(
        self,
        bronze_path: str,
        silver_path: str,
        dedup_keys: List[str],
        watermark_col: str,
        transformations: Optional[List[Callable[[DataFrame], DataFrame]]] = None,
    ) -> None:
        """
        Transform bronze to silver with cleaning, deduplication, and validation.

        Args:
            bronze_path: Path to bronze table
            silver_path: Path to silver table
            dedup_keys: Columns to use for deduplication
            watermark_col: Column to use for watermark/ordering
            transformations: List of transformation functions to apply
        """
        # Read bronze data
        bronze_df = self.spark.read.format("delta").load(bronze_path)

        # Get watermark for incremental processing
        if DeltaTable.isDeltaTable(self.spark, silver_path):
            silver_table = DeltaTable.forPath(self.spark, silver_path)
            max_watermark = (
                silver_table.toDF().agg(max(watermark_col)).collect()[0][0]
            )
            if max_watermark:
                bronze_df = bronze_df.filter(col(watermark_col) > max_watermark)

        # Apply transformations
        df = bronze_df
        if transformations:
            for transform in transformations:
                df = transform(df)

        # Skip if no new data
        if df.rdd.isEmpty():
            return

        # Deduplicate (keep latest by watermark)
        window = Window.partitionBy(dedup_keys).orderBy(col(watermark_col).desc())
        df_deduped = (
            df.withColumn("_row_num", row_number().over(window))
            .filter(col("_row_num") == 1)
            .drop("_row_num")
        )

        # Remove bronze metadata columns for silver
        silver_cols = [
            c for c in df_deduped.columns if not c.startswith("_")
        ] + [watermark_col]
        df_clean = df_deduped.select(*silver_cols)

        # Merge into silver (upsert)
        if DeltaTable.isDeltaTable(self.spark, silver_path):
            silver_table = DeltaTable.forPath(self.spark, silver_path)

            merge_condition = " AND ".join(
                [f"target.{key} = source.{key}" for key in dedup_keys]
            )

            (
                silver_table.alias("target")
                .merge(df_clean.alias("source"), merge_condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            # Initial load
            df_clean.write.format("delta").mode("overwrite").save(silver_path)

    def silver_to_gold(
        self,
        silver_tables: Dict[str, str],
        gold_path: str,
        aggregation_query: str,
        z_order_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Build gold layer aggregations from silver tables.

        Args:
            silver_tables: Dict mapping table name to path
            gold_path: Path to gold table
            aggregation_query: SQL query for aggregation
            z_order_cols: Columns to Z-order for optimization
        """
        # Register silver tables as temp views
        for name, path in silver_tables.items():
            self.spark.read.format("delta").load(path).createOrReplaceTempView(name)

        # Execute aggregation query
        gold_df = self.spark.sql(aggregation_query)

        # Write to gold
        gold_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(gold_path)

        # Optimize with Z-ordering
        if z_order_cols:
            gold_table = DeltaTable.forPath(self.spark, gold_path)
            gold_table.optimize().zOrderBy(z_order_cols).executeCompaction()

    def read_table(
        self,
        path: str,
        version: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> DataFrame:
        """
        Read a Delta table with optional time travel.

        Args:
            path: Path to Delta table
            version: Optional version number
            timestamp: Optional timestamp string

        Returns:
            DataFrame with table data
        """
        reader = self.spark.read.format("delta")

        if version is not None:
            reader = reader.option("versionAsOf", version)
        elif timestamp:
            reader = reader.option("timestampAsOf", timestamp)

        return reader.load(path)

    def optimize_table(
        self,
        path: str,
        partition_filter: Optional[str] = None,
        z_order_by: Optional[List[str]] = None,
    ) -> Dict:
        """
        Optimize a Delta table with optional Z-ordering.

        Args:
            path: Path to Delta table
            partition_filter: Optional partition filter
            z_order_by: Optional columns to Z-order

        Returns:
            Optimization metrics
        """
        table = DeltaTable.forPath(self.spark, path)
        optimize_builder = table.optimize()

        if partition_filter:
            optimize_builder = optimize_builder.where(partition_filter)

        if z_order_by:
            metrics = optimize_builder.zOrderBy(z_order_by).executeCompaction()
        else:
            metrics = optimize_builder.executeCompaction()

        return metrics

    def vacuum_table(self, path: str, retention_hours: int = 168) -> None:
        """
        Vacuum a Delta table to remove old files.

        Args:
            path: Path to Delta table
            retention_hours: Number of hours to retain old files
        """
        table = DeltaTable.forPath(self.spark, path)
        table.vacuum(retention_hours)

    def get_table_history(self, path: str, limit: Optional[int] = None) -> DataFrame:
        """
        Get the history of a Delta table.

        Args:
            path: Path to Delta table
            limit: Optional limit on number of history entries

        Returns:
            DataFrame with table history
        """
        table = DeltaTable.forPath(self.spark, path)
        if limit:
            return table.history(limit)
        return table.history()

    def restore_table(self, path: str, version: int) -> None:
        """
        Restore a Delta table to a previous version.

        Args:
            path: Path to Delta table
            version: Version to restore to
        """
        table = DeltaTable.forPath(self.spark, path)
        table.restoreToVersion(version)

    def apply_scd_type2(
        self,
        silver_path: str,
        updates_df: DataFrame,
        key_columns: List[str],
        track_columns: List[str],
        effective_date_col: str = "effective_date",
        end_date_col: str = "end_date",
        current_flag_col: str = "is_current",
    ) -> None:
        """
        Apply SCD Type 2 logic to a dimension table.

        This method implements Slowly Changing Dimension Type 2, which:
        - Preserves historical data by creating new records for changes
        - Tracks validity periods with effective/end dates
        - Maintains a current record flag

        Args:
            silver_path: Path to silver dimension table
            updates_df: DataFrame with new/updated records
            key_columns: Business key columns for matching
            track_columns: Columns to track for changes
            effective_date_col: Column name for effective date
            end_date_col: Column name for end date
            current_flag_col: Column name for current record flag
        """
        # Create hash of tracked columns for change detection
        hash_col = "_track_hash"
        updates_with_hash = updates_df.withColumn(
            hash_col, sha2(concat_ws("||", *[col(c) for c in track_columns]), 256)
        )

        # Add SCD columns to updates
        updates_with_scd = (
            updates_with_hash
            .withColumn(effective_date_col, current_date())
            .withColumn(end_date_col, lit(None).cast("date"))
            .withColumn(current_flag_col, lit(True))
        )

        # Check if target table exists
        if not DeltaTable.isDeltaTable(self.spark, silver_path):
            # Initial load - just write the data
            updates_with_scd.drop(hash_col).write.format("delta").mode(
                "overwrite"
            ).save(silver_path)
            return

        # Read existing silver table
        silver_table = DeltaTable.forPath(self.spark, silver_path)
        silver_df = silver_table.toDF()

        # Add hash to existing records
        silver_with_hash = silver_df.withColumn(
            hash_col, sha2(concat_ws("||", *[col(c) for c in track_columns]), 256)
        )

        # Build key match condition
        key_condition = " AND ".join(
            [f"target.{k} = source.{k}" for k in key_columns]
        )

        # Merge with SCD Type 2 logic
        # When matched and tracked columns changed: close old record
        # When not matched: insert as new current record
        (
            silver_table.alias("target")
            .merge(
                updates_with_scd.alias("source"),
                key_condition
            )
            # Close existing current record if attributes changed
            .whenMatchedUpdate(
                condition=f"""
                    target.{current_flag_col} = true AND
                    target.{hash_col} != source.{hash_col}
                """,
                set={
                    end_date_col: current_date(),
                    current_flag_col: "false",
                }
            )
            # Insert new records (both new keys and changed records)
            .whenNotMatchedInsertAll()
            .execute()
        )

        # Insert new versions for changed records
        # Find records that were closed but need new versions
        changed_records = (
            silver_df.alias("old")
            .join(
                updates_with_scd.alias("new"),
                [col(f"old.{k}") == col(f"new.{k}") for k in key_columns],
                "inner"
            )
            .where(
                (col(f"old.{current_flag_col}") == False) &
                (col(f"old.{hash_col}") != col(f"new.{hash_col}"))
            )
            .select([col(f"new.{c}") for c in updates_with_scd.columns])
        )

        # Append new versions if any
        if changed_records.count() > 0:
            changed_records.drop(hash_col).write.format("delta").mode(
                "append"
            ).save(silver_path)

    def silver_transformation(
        self,
        df: DataFrame,
        transformations: List[Callable[[DataFrame], DataFrame]],
    ) -> DataFrame:
        """
        Apply a series of transformations to a DataFrame.

        Args:
            df: Input DataFrame
            transformations: List of transformation functions

        Returns:
            Transformed DataFrame
        """
        result = df
        for transform in transformations:
            result = transform(result)
        return result

    def silver_deduplication(
        self,
        df: DataFrame,
        key_columns: List[str],
        order_column: str,
        ascending: bool = False,
    ) -> DataFrame:
        """
        Remove duplicates keeping the latest record per key.

        Args:
            df: Input DataFrame
            key_columns: Columns defining uniqueness
            order_column: Column to order by for keeping latest
            ascending: If True, keep first; if False, keep last

        Returns:
            Deduplicated DataFrame
        """
        order_col = col(order_column).asc() if ascending else col(order_column).desc()
        window = Window.partitionBy(key_columns).orderBy(order_col)

        return (
            df.withColumn("_row_num", row_number().over(window))
            .filter(col("_row_num") == 1)
            .drop("_row_num")
        )

    def merge_delta_tables(
        self,
        target_path: str,
        source_df: DataFrame,
        merge_keys: List[str],
        update_columns: Optional[List[str]] = None,
        delete_condition: Optional[str] = None,
    ) -> Dict:
        """
        Perform a Delta MERGE operation.

        Args:
            target_path: Path to target Delta table
            source_df: Source DataFrame with updates
            merge_keys: Columns to use for matching
            update_columns: Columns to update (None = all columns)
            delete_condition: Optional condition for deletes

        Returns:
            Merge metrics
        """
        if not DeltaTable.isDeltaTable(self.spark, target_path):
            source_df.write.format("delta").mode("overwrite").save(target_path)
            return {"inserted": source_df.count(), "updated": 0, "deleted": 0}

        target_table = DeltaTable.forPath(self.spark, target_path)
        merge_condition = " AND ".join(
            [f"target.{k} = source.{k}" for k in merge_keys]
        )

        merge_builder = (
            target_table.alias("target")
            .merge(source_df.alias("source"), merge_condition)
        )

        if update_columns:
            update_set = {c: f"source.{c}" for c in update_columns}
            merge_builder = merge_builder.whenMatchedUpdate(set=update_set)
        else:
            merge_builder = merge_builder.whenMatchedUpdateAll()

        if delete_condition:
            merge_builder = merge_builder.whenMatchedDelete(condition=delete_condition)

        merge_builder = merge_builder.whenNotMatchedInsertAll()
        merge_builder.execute()

        return {"merge_completed": True}

    def create_business_view(
        self,
        view_name: str,
        silver_tables: Dict[str, str],
        query: str,
    ) -> None:
        """
        Create a business view from silver tables.

        Args:
            view_name: Name for the view
            silver_tables: Dict mapping table aliases to paths
            query: SQL query defining the view
        """
        # Register silver tables as temp views
        for alias, path in silver_tables.items():
            self.spark.read.format("delta").load(path).createOrReplaceTempView(alias)

        # Create the business view
        self.spark.sql(f"CREATE OR REPLACE TEMP VIEW {view_name} AS {query}")
