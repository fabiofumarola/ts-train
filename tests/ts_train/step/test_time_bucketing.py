import pytest
from pydantic import ValidationError
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from ts_train.step.time_bucketing import TimeBucketing  # type: ignore
from ts_train.common.utils import is_column_window  # type: ignore


# TESTING INIT METHOD


@pytest.mark.parametrize("time_bucket_size", [-2, "2", "-2", 2.0, 0])
def test_init_wrong_time_bucket_size(time_bucket_size):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    with pytest.raises(ValidationError):
        TimeBucketing(
            time_column_name="DATA_TRANSAZIONE",
            time_bucket_size=time_bucket_size,
            time_bucket_granularity="days",
            time_bucket_col_name="bucket",
        )


def test_init_wrong_time_bucket_granularity():
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_granularity is not of an allowed value.
    """
    with pytest.raises(ValidationError):
        TimeBucketing(
            time_column_name="DATA_TRANSAZIONE",
            time_bucket_size=2,
            time_bucket_granularity="typo",
            time_bucket_col_name="bucket",
        )


# TESTING _PREPROCESS METHOD


def test_preprocess_emptiness(sample_dataframe_empty, standard_time_bucketing):
    """
    Tests if _preprocess method raises ValueError("Empty DataFrame") in case you provide
    an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_time_bucketing._preprocess(sample_dataframe_empty)


def test_preprocess_time_column_not_existing(
    sample_dataframe_01, standard_time_bucketing
):
    """
    Tests if _preprocess method raises ValueError("Column {time_column_name} is not a
    column) in the case we give a column name not present in the DataFrame.
    """
    standard_time_bucketing.time_column_name = "not_existing_column_name"

    with pytest.raises(
        ValueError,
        match=f"Column {standard_time_bucketing.time_column_name} is not a column",
    ):
        standard_time_bucketing._preprocess(sample_dataframe_01)


def test_preprocess_time_column_not_timestamp(
    sample_dataframe_01, standard_time_bucketing
):
    """Tests if _preprocess method raises ValueError("Column {time_column_name} not a
    timestamp column) in the case we give the method a column which is not a time.
    """
    standard_time_bucketing.time_column_name = "CA_CATEGORY_LIV0"

    with pytest.raises(
        ValueError,
        match=(
            f"Column {standard_time_bucketing.time_column_name} is not a timestamp"
            " column"
        ),
    ):
        standard_time_bucketing._preprocess(sample_dataframe_01)


def test_preprocess_time_bucket_col_name_existing(
    sample_dataframe_01, standard_time_bucketing
):
    """Tests if _preprocess method raises ValueError("Column {self.time_bucket_col_name}
    already a column name) in the case we give the method a column which is already an
    existing column name.
    """
    standard_time_bucketing.time_bucket_col_name = "CA_CATEGORY_LIV0"

    with pytest.raises(
        ValueError,
        match=(
            f"Column {standard_time_bucketing.time_bucket_col_name} is already a column"
        ),
    ):
        standard_time_bucketing._preprocess(sample_dataframe_01)


# Testing _PROCESS METHOD


def test_process_new_column_added(sample_dataframe_01, spark, standard_time_bucketing):
    """Tests if _process method adds a new column."""
    result_df = standard_time_bucketing._process(sample_dataframe_01, spark)

    assert len(sample_dataframe_01.columns) + 1 == len(result_df.columns)


def test_process_new_column_type(
    sample_dataframe_01, spark, standard_time_bucketing: TimeBucketing
):
    """Tests if _process method adds a new column with the right datactype."""
    result_df = standard_time_bucketing._process(sample_dataframe_01, spark)

    assert is_column_window(result_df, standard_time_bucketing.time_bucket_col_name)


def test_process_timestamps_inside_bucket(
    sample_dataframe_01, spark, standard_time_bucketing
):
    """Tests if _process method groups rows in the right way."""
    result_df = standard_time_bucketing._process(sample_dataframe_01, spark)

    def is_inside_time_bucket(timestamp, bucket):
        return bucket.start <= timestamp < bucket.end

    is_inside_time_bucket_udf = F.udf(
        lambda timestamp, bucket: is_inside_time_bucket(timestamp, bucket),
        BooleanType(),
    )

    result_df = result_df.withColumn(
        "is_inside_time_bucket",
        is_inside_time_bucket_udf(
            F.col(standard_time_bucketing.time_column_name),
            F.col(standard_time_bucketing.time_bucket_col_name),
        ),
    )

    assert (
        result_df.count()
        == result_df.filter(
            F.col("is_inside_time_bucket") == True  # noqa: E712
        ).count()
    )
