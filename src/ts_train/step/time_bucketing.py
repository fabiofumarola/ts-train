from typing import *

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pydantic import BaseModel, StrictStr

from ts_train.step.core import AbstractPipelineStep
from ts_train.common.enums import TimeBucketGranularity
from ts_train.common.types import PositiveStrictInt
from ts_train.common.utils import *


# TODO add month, year aggregation capability
class TimeBucketing(AbstractPipelineStep, BaseModel):
    """
    Associate each row with a interval of time (called time bucket).

    It create a new colomn named time_bucket_col_name that is composed by a start and
    end timestamp.

    Attributes:
        time_zone (StrictStr): Time zone in the following format: [Area]/[City]. In the
            case of Italy it is: Europe/Rome.
        time_column_name (StrictStr): Column name corresponding to the column in which
            there are timestamp/date/datetime values.
        time_bucket_size (PositiveStrictInt): Column for expressing the bucket duration.
            Here you have to specify only the number. For example if you want a duration
            of 2 days, here you have to set: 2. It hase to be a greater than 0 value.
        time_bucket_granularity (TimeBucketGranularity): Column for expressing the
            bucket unit. Here you have to specify only the unit. For example if you want
            a duration of 2 days, here you have to set: "days". Allowed values are:
            week, weeks, day, days, hour, hours, minute, minutes.
        time_bucket_col_name (StrictStr): Column name corresponding to the column in
            which you have to add time buckets. It should be a column name not already
            used in the DataFrame.
    """

    time_zone: StrictStr
    time_column_name: StrictStr
    time_bucket_size: PositiveStrictInt
    time_bucket_granularity: TimeBucketGranularity
    time_bucket_col_name: StrictStr

    def _preprocess(self, df: DataFrame) -> None:
        """Validates every condition of the DataFrame provided and of the instance
        attributes which are dependent on the DataFrame.

        Args:
            df (DataFrame): DataFrame to check

        Raises:
            ValueError: with "Empty DataFrame" message if the DataFrame is empty
            ValueError: with "Column {time_column_name} is not a column" if the
                provided time_column_name is not present in the provided DataFrame
            ValueError: with "Column {time_column_name} not a timestamp column" if
                the provided time_column_name is not a timestamp/date column
            ValueError: with "Column {time_bucket_col_name} already a column name"
                if time_bucket_col_name is already a column of the provided DataFrame

        Returns:
            None: if it returns None it means no issues have been found
        """

        # Checks if the DataFrame is full or empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        # Checks if time_column_name is a column and is a timestamp column
        if not is_column_present(df, self.time_column_name):
            raise ValueError(f"Column {self.time_column_name} is not a column")

        if not is_column_timestamp(df, self.time_column_name):
            raise ValueError(
                f"Column {self.time_column_name} is not a timestamp column"
            )

        # Checks if time_bucket_col_name is not already an existing column name
        if is_column_present(df, self.time_bucket_col_name):
            raise ValueError(f"Column {self.time_bucket_col_name} is already a column")

        return None

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """Creates a new column with time_bucket_column name with inside time buckets.

        It sets the Spark Time Zone to the provided time_zone.

        Args:
            df (DataFrame): DataFrame on which to perform the operation.
            spark (SparkSession): SparkSession to utilize.

        Returns:
            DataFrame: DataFrame with added column with time_bucket_column column name.
        """
        spark.conf.set("spark.sql.session.timeZone", self.time_zone)

        time_bucket_duration = (
            str(self.time_bucket_size) + " " + self.time_bucket_granularity
        )
        time_bucket_column = F.window(
            timeColumn=F.col(self.time_column_name),
            windowDuration=time_bucket_duration,
        )

        df = df.withColumn(self.time_bucket_col_name, time_bucket_column)

        return df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        return result
