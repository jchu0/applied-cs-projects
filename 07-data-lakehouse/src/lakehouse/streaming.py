"""Streaming data processing for the lakehouse."""

from typing import Callable, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    from_json,
    lit,
    to_date,
    window,
)
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import StructType


class StreamingProcessor:
    """Process streaming data into the lakehouse."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def stream_from_kafka(
        self,
        bootstrap_servers: str,
        topic: str,
        schema: StructType,
        group_id: str = "lakehouse-consumer",
        starting_offsets: str = "latest",
    ) -> DataFrame:
        """
        Create a streaming DataFrame from Kafka.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic to consume
            schema: Schema of the message value
            group_id: Consumer group ID
            starting_offsets: Where to start reading (latest, earliest)

        Returns:
            Streaming DataFrame
        """
        return (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", bootstrap_servers)
            .option("subscribe", topic)
            .option("startingOffsets", starting_offsets)
            .option("kafka.group.id", group_id)
            .load()
            .select(
                from_json(col("value").cast("string"), schema).alias("data"),
                col("timestamp").alias("kafka_timestamp"),
            )
            .select("data.*", "kafka_timestamp")
        )

    def stream_to_bronze(
        self,
        df: DataFrame,
        bronze_path: str,
        source_name: str,
        checkpoint_path: str,
        trigger_interval: str = "1 minute",
        watermark_col: Optional[str] = None,
        watermark_delay: str = "5 minutes",
    ) -> StreamingQuery:
        """
        Stream data to bronze layer.

        Args:
            df: Streaming DataFrame
            bronze_path: Path to bronze table
            source_name: Name of the data source
            checkpoint_path: Path for streaming checkpoints
            trigger_interval: Processing interval
            watermark_col: Optional column for watermarking
            watermark_delay: How late data can arrive

        Returns:
            StreamingQuery handle
        """
        # Add bronze metadata
        df_bronze = (
            df.withColumn("_source", lit(source_name))
            .withColumn("_ingestion_ts", current_timestamp())
            .withColumn("_ingestion_date", to_date(current_timestamp()))
        )

        # Add watermark if specified
        if watermark_col:
            df_bronze = df_bronze.withWatermark(watermark_col, watermark_delay)

        # Write to bronze
        return (
            df_bronze.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
            .partitionBy("_ingestion_date")
            .start(bronze_path)
        )

    def stream_aggregation(
        self,
        df: DataFrame,
        output_path: str,
        checkpoint_path: str,
        group_cols: List[str],
        agg_exprs: Dict[str, str],
        window_col: str,
        window_duration: str,
        slide_duration: Optional[str] = None,
        watermark_delay: str = "10 minutes",
        trigger_interval: str = "1 minute",
    ) -> StreamingQuery:
        """
        Stream windowed aggregations.

        Args:
            df: Streaming DataFrame
            output_path: Path to output table
            checkpoint_path: Path for streaming checkpoints
            group_cols: Columns to group by
            agg_exprs: Aggregation expressions {alias: expr}
            window_col: Timestamp column for windowing
            window_duration: Window size (e.g., "1 hour")
            slide_duration: Sliding interval (None for tumbling)
            watermark_delay: How late data can arrive
            trigger_interval: Processing interval

        Returns:
            StreamingQuery handle
        """
        # Add watermark
        df_watermarked = df.withWatermark(window_col, watermark_delay)

        # Create window expression
        if slide_duration:
            window_expr = window(col(window_col), window_duration, slide_duration)
        else:
            window_expr = window(col(window_col), window_duration)

        # Build aggregation
        agg_df = df_watermarked.groupBy(
            window_expr.alias("window"), *group_cols
        )

        # Apply aggregations
        from pyspark.sql.functions import expr
        agg_columns = [
            expr(agg_expr).alias(alias)
            for alias, agg_expr in agg_exprs.items()
        ]
        result_df = agg_df.agg(*agg_columns)

        # Flatten window struct
        result_df = (
            result_df.withColumn("window_start", col("window.start"))
            .withColumn("window_end", col("window.end"))
            .drop("window")
        )

        # Write to output
        return (
            result_df.writeStream.format("delta")
            .outputMode("update")
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
            .start(output_path)
        )

    def stream_with_state(
        self,
        df: DataFrame,
        output_path: str,
        checkpoint_path: str,
        state_func: Callable,
        group_cols: List[str],
        timeout_conf: str = "NoTimeout",
        trigger_interval: str = "1 minute",
    ) -> StreamingQuery:
        """
        Stream with arbitrary stateful processing.

        Args:
            df: Streaming DataFrame
            output_path: Path to output table
            checkpoint_path: Path for streaming checkpoints
            state_func: Function for state processing
            group_cols: Columns to group state by
            timeout_conf: Timeout configuration
            trigger_interval: Processing interval

        Returns:
            StreamingQuery handle
        """
        from pyspark.sql.streaming import GroupState, GroupStateTimeout

        timeout = {
            "NoTimeout": GroupStateTimeout.NoTimeout,
            "ProcessingTimeTimeout": GroupStateTimeout.ProcessingTimeTimeout,
            "EventTimeTimeout": GroupStateTimeout.EventTimeTimeout,
        }.get(timeout_conf, GroupStateTimeout.NoTimeout)

        result_df = df.groupBy(group_cols).applyInPandasWithState(
            state_func,
            outputStructType=df.schema,
            stateStructType=df.schema,
            outputMode="update",
            timeoutConf=timeout,
        )

        return (
            result_df.writeStream.format("delta")
            .outputMode("update")
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
            .start(output_path)
        )

    def stream_dedupe(
        self,
        df: DataFrame,
        output_path: str,
        checkpoint_path: str,
        id_columns: List[str],
        watermark_col: str,
        watermark_delay: str = "30 minutes",
        trigger_interval: str = "1 minute",
    ) -> StreamingQuery:
        """
        Stream with deduplication.

        Args:
            df: Streaming DataFrame
            output_path: Path to output table
            checkpoint_path: Path for streaming checkpoints
            id_columns: Columns that identify duplicates
            watermark_col: Timestamp column for watermarking
            watermark_delay: How long to remember IDs
            trigger_interval: Processing interval

        Returns:
            StreamingQuery handle
        """
        # Add watermark
        df_watermarked = df.withWatermark(watermark_col, watermark_delay)

        # Deduplicate
        df_deduped = df_watermarked.dropDuplicates(id_columns)

        return (
            df_deduped.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
            .start(output_path)
        )

    def monitor_query(self, query: StreamingQuery) -> Dict:
        """
        Get monitoring information for a streaming query.

        Args:
            query: StreamingQuery to monitor

        Returns:
            Dict with query status and progress
        """
        return {
            "id": str(query.id),
            "name": query.name,
            "is_active": query.isActive,
            "status": query.status,
            "last_progress": query.lastProgress,
            "recent_progress": query.recentProgress,
        }


class ChangeDataFeedProcessor:
    """Process Change Data Feed for incremental updates."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def enable_cdf(self, table_path: str) -> None:
        """
        Enable Change Data Feed on a table.

        Args:
            table_path: Path to Delta table
        """
        self.spark.sql(f"""
            ALTER TABLE delta.`{table_path}`
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

    def read_changes(
        self,
        table_path: str,
        start_version: Optional[int] = None,
        end_version: Optional[int] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
    ) -> DataFrame:
        """
        Read changes from Change Data Feed.

        Args:
            table_path: Path to Delta table
            start_version: Starting version
            end_version: Ending version
            start_timestamp: Starting timestamp
            end_timestamp: Ending timestamp

        Returns:
            DataFrame with changes
        """
        reader = self.spark.read.format("delta").option("readChangeFeed", "true")

        if start_version is not None:
            reader = reader.option("startingVersion", start_version)
        if end_version is not None:
            reader = reader.option("endingVersion", end_version)
        if start_timestamp:
            reader = reader.option("startingTimestamp", start_timestamp)
        if end_timestamp:
            reader = reader.option("endingTimestamp", end_timestamp)

        return reader.load(table_path)

    def stream_changes(
        self,
        table_path: str,
        start_version: Optional[int] = None,
    ) -> DataFrame:
        """
        Stream changes from Change Data Feed.

        Args:
            table_path: Path to Delta table
            start_version: Starting version

        Returns:
            Streaming DataFrame with changes
        """
        reader = (
            self.spark.readStream.format("delta")
            .option("readChangeFeed", "true")
        )

        if start_version is not None:
            reader = reader.option("startingVersion", start_version)

        return reader.load(table_path)

    def propagate_changes(
        self,
        source_path: str,
        target_path: str,
        merge_keys: List[str],
        checkpoint_path: str,
        start_version: int = 0,
    ) -> StreamingQuery:
        """
        Propagate changes from source to target table.

        Args:
            source_path: Path to source Delta table
            target_path: Path to target Delta table
            merge_keys: Columns to use for merge
            checkpoint_path: Path for streaming checkpoints
            start_version: Starting version

        Returns:
            StreamingQuery handle
        """
        changes = self.stream_changes(source_path, start_version)

        def process_batch(batch_df: DataFrame, batch_id: int):
            """Process a micro-batch of changes."""
            from delta import DeltaTable

            if batch_df.isEmpty():
                return

            # Separate by change type
            inserts = batch_df.filter("_change_type = 'insert'")
            updates = batch_df.filter("_change_type = 'update_postimage'")
            deletes = batch_df.filter("_change_type = 'delete'")

            # Get target table
            target = DeltaTable.forPath(self.spark, target_path)

            # Process inserts and updates
            upserts = inserts.union(updates).drop(
                "_change_type", "_commit_version", "_commit_timestamp"
            )

            if not upserts.isEmpty():
                merge_condition = " AND ".join([
                    f"target.{key} = source.{key}" for key in merge_keys
                ])

                (
                    target.alias("target")
                    .merge(upserts.alias("source"), merge_condition)
                    .whenMatchedUpdateAll()
                    .whenNotMatchedInsertAll()
                    .execute()
                )

            # Process deletes
            if not deletes.isEmpty():
                delete_condition = " OR ".join([
                    f"({' AND '.join([f'{key} = {row[key]}' for key in merge_keys])})"
                    for row in deletes.collect()
                ])
                target.delete(delete_condition)

        return (
            changes.writeStream
            .foreachBatch(process_batch)
            .option("checkpointLocation", checkpoint_path)
            .start()
        )


# Alias for backward compatibility with tests
StreamProcessor = StreamingProcessor
