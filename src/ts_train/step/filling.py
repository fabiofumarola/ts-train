from typing import *

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pydantic import BaseModel, StrictStr

from ts_train.step.core import AbstractPipelineStep
from ts_train.common.enums import TimeBucketGranularity
from ts_train.common.types import PositiveStrictInt
from ts_train.common.utils import *


class Filling(AbstractPipelineStep, BaseModel):
    time_bucket_col_name: StrictStr
    identifier_cols_name: Union[StrictStr, List[StrictStr]]
    time_bucket_size: PositiveStrictInt
    time_bucket_granularity: TimeBucketGranularity

    def _preprocess(self, df: DataFrame) -> None:
        # Checks if the DataFrame is full or empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        # Checks if time_bucket_col_name is a column and is a timestamp column
        if not is_column_present(df, self.time_bucket_col_name):
            raise ValueError(f"Column {self.time_bucket_col_name} is not a column")

        if not is_column_window(df, self.time_bucket_col_name):
            raise ValueError(
                f"Column {self.time_bucket_col_name} is not a window column"
            )

        # Convert identifier_cols_name into List[StrictStr]
        if type(self.identifier_cols_name) == str:
            self.identifier_cols_name = [self.identifier_cols_name]

        # Checks if identifier_cols_name are columns
        for identifier_col_name in self.identifier_cols_name:
            if not is_column_present(df, identifier_col_name):
                raise ValueError(f"Column {identifier_col_name} is not a column")

        return None

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        # Creates a list of identifier columns
        identifier_cols = [
            F.col(identifier_col_name)
            for identifier_col_name in self.identifier_cols_name
        ]

        # Creates the bucket size
        time_bucket_duration = (
            str(self.time_bucket_size) + " " + self.time_bucket_granularity
        )

        # Creates aliases for simplicity and code readability
        time_bucket_start = f"{self.time_bucket_col_name}_start"
        time_bucket_end = f"{self.time_bucket_col_name}_end"
        min_time_bucket_start = f"min_{self.time_bucket_col_name}_start"
        max_time_bucket_end = f"max_{self.time_bucket_col_name}_end"

        # Creates a new DataFrame with only the identifier columns
        # Splits the bucket into two column, start and end assigning to new columns
        ids_df = df.select(
            *identifier_cols,
            F.col(self.time_bucket_col_name).start.alias(time_bucket_start),
            F.col(self.time_bucket_col_name).end.alias(time_bucket_end),
        )

        # Takes only one record for every user
        # Saves only the min start and the max end
        ids_df = ids_df.groupBy(*identifier_cols).agg(
            F.min(time_bucket_start).alias(min_time_bucket_start),
            F.max(time_bucket_end).alias(max_time_bucket_end),
        )

        # Creates a new column with inside for each user an array of timestamps from
        # the min to the max of the time bucket of that particular user
        # Drops min and max columns
        ids_timestamps_df = ids_df.withColumn(
            "timestamps",
            F.expr(
                f"sequence(to_timestamp({min_time_bucket_start}),"
                f" to_timestamp({max_time_bucket_end}), interval"
                f" {time_bucket_duration})"
            ),
        ).drop(
            min_time_bucket_start,
            max_time_bucket_end,
        )

        # Explodes the array of timestamps into a series of rows each with a timestamp
        # column representing the start of that time bucket
        # Drops timestamps array column
        ids_timestamps_df = ids_timestamps_df.withColumn(
            "timestamp", F.explode(F.col("timestamps"))  # TODO make timestamp param
        ).drop(
            "timestamps",
        )

        df = df.withColumn("timestamp", F.col(self.time_bucket_col_name).start)

        # Joins the DataFrame with the new DataFrame in which has been generated
        # timestamps for every user from its min timestamp to his max
        # Fills with 0 null values of every column
        # Drops time bucket column
        join_on_cols = [*self.identifier_cols_name, "timestamp"]
        df = (
            df.join(ids_timestamps_df, on=join_on_cols, how="right")
            .fillna(0)  # TODO verify this has no negative effect
            .drop(self.time_bucket_col_name)  # TODO choose if we want to drop
        )

        return df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        return result
