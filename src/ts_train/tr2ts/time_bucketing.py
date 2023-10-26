from typing import Tuple, Literal

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.sql import functions as F

import pandas as pd
from pandas.tseries.offsets import DateOffset

from pydantic import BaseModel, StrictStr
from pyspark.sql.functions import min, max

from ts_train.tr2ts.core import AbstractPipelineStep
from ts_train.common.types import PositiveStrictInt
from ts_train.common.utils import (
    is_dataframe_empty,
    is_column_present,
    is_column_timestamp,
)


def get_data_offset(time_bucket_size: int, time_bucket_granularity: str) -> DateOffset:
    """Get the offset date for the provided size and granularity.

    Args:
        time_bucket_size (int): Duration for each bucket (numeric value).
        time_bucket_granularity (str): Unit of time for bucket size.

    Raises:
        ValueError: with message "Granularity {time_bucket_granularity} not supported"
            when you ask for a not supported granualrity.

    Returns:
        DateOffset: Offset for the provided size and granularity.
    """
    granularity = time_bucket_granularity[0].upper()

    if granularity == "H":
        return DateOffset(hours=time_bucket_size)
    elif granularity == "D":
        return DateOffset(days=time_bucket_size)
    elif granularity == "W":
        return DateOffset(weeks=time_bucket_size)
    elif granularity == "M":
        return DateOffset(months=time_bucket_size)
    elif granularity == "Y":
        return DateOffset(years=time_bucket_size)
    else:
        raise ValueError(f"Granularity {time_bucket_granularity} not supported")


class TimeBucketing(AbstractPipelineStep, BaseModel):
    """
    Associate each row with a time interval (time bucket).
    It creates new columns named bucket_start and bucket_end.

    Attributes:
        time_col_name (StrictStr): Column name containing timestamp/date values.
        time_bucket_size (PositiveStrictInt): Duration for each bucket (numeric value).
        time_bucket_granularity (Literal[str]): Unit of time for bucket size. Possible
            values: hour, hours, day, days, week, weeks, month, months, year, years".
    """

    time_col_name: StrictStr
    time_bucket_size: PositiveStrictInt
    time_bucket_granularity: Literal[
        "hour",
        "hours",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
    ]

    def _preprocess(self, df: DataFrame) -> None:
        """Checks if the provided DataFrame and other parameters are valid. Raises
        exceptions otherwise.

        Args:
            df (DataFrame): DataFrame to check

        Raises:
            ValueError: with message "Empty DataFrame" if you provide a DataFrame with
                no data inside.
            ValueError: with message "Column {self.time_col_name} is not present" when
                the column with name time_col_name is not present in the DataFrame.
            ValueError: with message "Column {self.time_col_name} is not a timestamp
                column" when the time_col_name column is not a timestamp column.
        """
        # Fix for different versions of Pandas
        if not hasattr(df, "iteritems"):
            pd.DataFrame.iteritems = pd.DataFrame.items  # type: ignore

        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        if not is_column_present(df, self.time_col_name):
            raise ValueError(f"Column {self.time_col_name} is not present")

        if not is_column_timestamp(df, self.time_col_name):
            raise ValueError(f"Column {self.time_col_name} is not a timestamp column")

    def _create_timeline(
        self, df: DataFrame
    ) -> Tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
        """Extract the minimum and maximum date from the DataFrame and generate a
        timeline that is a list of dates with the provided size and granularity that
        goes from the minimum date to the maximum date.

        Args:
            df (DataFrame): DataFrame containing data.

        Raises:
            ValueError: with "Empty DataFrame" if the DataFrame is empty.

        Returns:
            timeline (pd.DatetimeIndex): Generated timeline with pandas daterange().
            min_date (pd.Timestamp): Minimum date in the DataFrame.
            max_date (pd.Timestamp): Maximum date in the DataFrame with offset.
        """
        first_element = df.select(min(self.time_col_name)).collect()[0][0]

        if first_element:
            min_date = pd.to_datetime(str(first_element))
            min_date = min_date.to_period(
                self.time_bucket_granularity[0]
            ).to_timestamp()

            # Calculate max_date as last element's date + offset to ensure inclusivity
            # of the maximum date in the last bucket
            max_date = pd.to_datetime(
                df.select(max(self.time_col_name)).collect()[0][0]
            )

            granularity = self.time_bucket_granularity[0].upper()

            if granularity == "M":
                frequency = f"{self.time_bucket_size}MS"
            elif granularity == "W":
                frequency = f"{self.time_bucket_size}W-MON"
            elif granularity == "Y":
                frequency = f"{self.time_bucket_size}YS"
            else:
                frequency = f"{self.time_bucket_size}{granularity}"

            timeline = pd.date_range(min_date, max_date, freq=frequency)

            return timeline, min_date, max_date
        else:
            raise ValueError("Empty DataFrame")

    def _create_df_with_buckets(
        self, spark_session: SparkSession, date_range: pd.DatetimeIndex
    ) -> DataFrame:
        """Turn the pandas timeline into a Spark DataFrame were the timeline event are
        splitted in buckets with start and end. In this process is critical to avoid
        that spark automaticly change date due to timezone or daylight saving time.

        Args:
            spark_session (SparkSession): Spark session.
            date_range (pd.DatetimeIndex): Generated date range.

        Returns:
            DataFrame: DataFrame containing buckets with start and end.
        """
        offset = get_data_offset(self.time_bucket_size, self.time_bucket_granularity)
        dates_df = pd.DataFrame(date_range, columns=["bucket_start"])
        dates_df["bucket_end"] = dates_df["bucket_start"].shift(-1)
        dates_df.at[dates_df.index[-1], "bucket_end"] = date_range[-1] + offset
        dates_df["bucket_start"] = dates_df["bucket_start"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        dates_df["bucket_end"] = dates_df["bucket_end"].dt.strftime("%Y-%m-%d %H:%M:%S")

        bucket_df = spark_session.createDataFrame(dates_df)

        return bucket_df

    def _bucketize_data(self, bucket_df: DataFrame, data_df: DataFrame) -> DataFrame:
        """
        Perform data bucketing based on the provided buckets.

        Args:
            bucket_df (DataFrame): DataFrame containing buckets with start and end.
            data_df (DataFrame): DataFrame containing data to be bucketized.

        Returns:
            final_df (DataFrame): DataFrame with bucketized data.
        """
        final_df = data_df.join(
            bucket_df,
            expr(
                f"{self.time_col_name} >= bucket_start AND {self.time_col_name} <"
                " bucket_end"
            ),
        )
        final_df = final_df.withColumn(
            "bucket_end", expr("bucket_end - interval 1 second")
        )
        final_df = final_df.withColumn(
            "bucket_start", F.to_timestamp(F.col("bucket_start"), "yyyy-MM-dd HH:mm:ss")
        )
        final_df = final_df.withColumn(
            "bucket_end", F.to_timestamp(F.col("bucket_end"), "yyyy-MM-dd HH:mm:ss")
        )

        return final_df

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Put each transaction date inside a timebucket and return the DataFrame with
        the new columns that indicate the bucket_start and bucket_end.

        Args:
            df (DataFrame): DataFrame to operate on.
            spark (SparkSession): SparkSession to use.

        Returns:
            DataFrame: DataFrame with added bucket_start and bucket_end columns.
        """
        timeline, _, _ = self._create_timeline(df)
        bucket_df = self._create_df_with_buckets(spark, timeline)
        final_df = self._bucketize_data(bucket_df, df)

        return final_df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        """
        Post-process the result DataFrame.

        Args:
            result (DataFrame): Resulting DataFrame from _process.

        Returns:
            DataFrame: Processed result DataFrame.
        """
        return result
