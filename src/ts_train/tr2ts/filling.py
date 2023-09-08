from typing import *

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pydantic import BaseModel, StrictStr
from pyspark.sql.functions import expr
from pyspark.sql.functions import col, date_format

from ts_train.tr2ts.core import AbstractPipelineStep
from ts_train.tr2ts.time_bucketing import TimeBucketing
from ts_train.common.utils import *


class Filling(AbstractPipelineStep, BaseModel):
    identifier_cols_name: List[StrictStr]
    time_bucket_step: TimeBucketing

    def _preprocess(self, df: DataFrame) -> None:
        # Checks if the DataFrame is full or empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        # Checks if identifier_cols_name are columns
        for identifier_col_name in self.identifier_cols_name:
            if not is_column_present(df, identifier_col_name):
                raise ValueError(f"Column {identifier_col_name} is not a column")

        return None

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        # Creo la nuova timeline per tutti in pandas
        self.time_bucket_step.time_col_name = "bucket_start"

        # Creates a list of identifier columns
        identifier_cols = [
            F.col(identifier_col_name)
            for identifier_col_name in self.identifier_cols_name
        ]

        # Creates aliases for simplicity and code readability
        time_bucket_start = "bucket_start_start"
        time_bucket_end = "bucket_start_end"
        min_time_bucket_start = "min_bucket_start"
        max_time_bucket_end = "max_bucket_start"

        # Creates a new DataFrame with only the identifier columns
        # Splits the bucket into two column, start and end assigning to new columns
        ids_df = df.select(
            *identifier_cols,
            F.col("bucket_start").alias(time_bucket_start),
            F.col("bucket_end").alias(time_bucket_end),
        )

        # Takes only one record for every user
        # Saves only the min start and the max end
        ids_df = ids_df.groupBy(*identifier_cols).agg(
            F.min(time_bucket_start).alias(min_time_bucket_start),
            F.max(time_bucket_end).alias(max_time_bucket_end),
        )

        # create the new timeline with every buckets
        df_to_timeline = df.withColumn(
            "bucket_start", date_format(col("bucket_start"), "yyyy-MM-dd HH:mm:ss")
        )
        df_to_timeline = df.withColumn(
            "bucket_end", date_format(col("bucket_end"), "yyyy-MM-dd HH:mm:ss")
        )

        timeline, _, _ = self.time_bucket_step._create_timeline(df_to_timeline)
        bucket_df = self.time_bucket_step._create_df_with_buckets(spark, timeline)
        bucket_df = bucket_df.withColumn(
            "bucket_end", expr("bucket_end - interval 1 second")
        )

        # Collego gli utenti alla nuova timeline
        # Converte le colonne delle date in tipo timestamp
        bucket_df = bucket_df.withColumn(
            "bucket_start", col("bucket_start").cast("timestamp")
        )
        bucket_df = bucket_df.withColumn(
            "bucket_end", col("bucket_end").cast("timestamp")
        )

        # Esegue la join basata sulla condizione di intervallo
        result_df = ids_df.join(
            bucket_df,
            (bucket_df["bucket_start"] >= ids_df[min_time_bucket_start])
            & (bucket_df["bucket_end"] <= ids_df[max_time_bucket_end]),
        )

        # Seleziona le colonne desiderate per la tabella finale
        all_timestamp_per_clients = result_df.select(
            *self.identifier_cols_name, "bucket_start", "bucket_end"
        )

        # Joins the DataFrame with the new DataFrame in which has been generated
        # timestamps for every user from its min timestamp to his max
        # Fills with 0 null values of every column
        # Drops time bucket column
        join_on_cols = self.identifier_cols_name + ["bucket_start", "bucket_end"]
        df = df.join(all_timestamp_per_clients, on=join_on_cols, how="right").fillna(0)

        df = df.orderBy(*join_on_cols)

        return df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        return result
