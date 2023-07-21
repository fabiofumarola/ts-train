import pytest

from pydantic import ValidationError
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ts_train.step.filling import Filling  # type: ignore


# TESTING INIT METHOD


@pytest.mark.parametrize("time_bucket_size", [-2, "2", "-2", 2.0, 0])
def test_init_wrong_time_bucket_size(time_bucket_size):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    with pytest.raises(ValidationError):
        Filling(
            time_bucket_col_name="bucket",
            identifier_cols_name="ID_BIC_CLIENTE",
            time_bucket_size=time_bucket_size,
            time_bucket_granularity="days",
        )


def test_init_wrong_time_bucket_granularity():
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_granularity is not of an allowed value.
    """
    with pytest.raises(ValidationError):
        Filling(
            time_bucket_col_name="bucket",
            identifier_cols_name="ID_BIC_CLIENTE",
            time_bucket_size=2,
            time_bucket_granularity="typo",
        )


# TESTING _PREPROCESS METHOD


def test_preprocess_emptiness(sample_dataframe_empty, standard_filling):
    """Tests if _preprocess method raises ValueError("Empty DataFrame") in case you
    provide an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_filling._preprocess(sample_dataframe_empty)


def test_preprocess_time_bucket_col_name_not_existing(
    sample_dataframe_02,
    standard_filling: Filling,
):
    """Tests if _preprocess method raises ValueError("Column {time_bucket_col_name} is
    not a column") in case the time_bucket_col_name column is not present."""
    with pytest.raises(
        ValueError,
        match=f"Column {standard_filling.time_bucket_col_name} is not a column",
    ):
        standard_filling._preprocess(sample_dataframe_02)


def test_preprocess_time_bucket_col_name_wrong_type(
    sample_dataframe_02,
    standard_filling: Filling,
):
    """Tests if _preprocess method raises ValueError("Column {time_bucket_col_name} is
    not a window column") in case the time_bucket_col_name column is not a window."""
    standard_filling.time_bucket_col_name = "DATA_TRANSAZIONE"

    with pytest.raises(
        ValueError,
        match=f"Column {standard_filling.time_bucket_col_name} is not a window column",
    ):
        standard_filling._preprocess(sample_dataframe_02)


@pytest.mark.parametrize(
    "identifier_cols_name", ["typo", ["typo"], ["typo1", "typo2"], ["typo", "bucket"]]
)
def test_preprocess_identifier_cols_name_not_existing(
    spark,
    sample_dataframe_02,
    standard_time_bucketing,
    standard_filling,
    identifier_cols_name,
):
    """Tests if _preprocess method raises ValueError("Column {column name} is not a
    column") in the case in which identifier_cols_name is a wrong column name or it is
    a list with inside a wrong column name."""
    standard_filling.identifier_cols_name = identifier_cols_name

    df = standard_time_bucketing(sample_dataframe_02, spark)

    if type(identifier_cols_name) == str:
        identifier_cols_name_list = [identifier_cols_name]
    else:
        identifier_cols_name_list = identifier_cols_name

    possible_identifier_cols_name_pattern = "|".join(identifier_cols_name_list)
    with pytest.raises(
        ValueError,
        match=r"Column ("
        + possible_identifier_cols_name_pattern
        + r") is not a column",
    ):
        standard_filling._preprocess(df)


# TESTING _PROCESS METHOD


def create_diff_dataframe(df, identifier_cols_name):
    """Given a DataFrame with timestamp column it creates a new DataFrame where there
    are only samples where the difference between one timestamp and the next one is
    different from 1 day."""

    # Converts timestamp column to unix timestamp a creates a new column
    df = df.withColumn("timestamp_unix", F.unix_timestamp("timestamp"))

    # Creates a window partitioning on identifiers, ordered by timestamp_unix
    identifier_cols = [
        F.col(identifier_col_name) for identifier_col_name in identifier_cols_name
    ]
    window_spec = Window.partitionBy(identifier_cols).orderBy("timestamp_unix")

    # Creates a difference between couples of rows
    df = df.withColumn(
        "timestamp_unix_diff",
        F.col("timestamp_unix") - F.lag(F.col("timestamp_unix"), 1).over(window_spec),
    )

    # Creates a DataFrame with only rows with diff different from the right one
    diff_df = df.filter(
        (F.col("timestamp_unix_diff") != 24 * 60 * 60)
        & F.col("timestamp_unix_diff").isNotNull()
    )

    return diff_df


def test_process_samples_timestamp_distance(
    spark, sample_dataframe_02, standard_time_bucketing, standard_filling
):
    """Tests that the difference between each sample for each id is one day."""

    # Sets time_bucket_size to 1 day. If you wan to change it remember to change also
    # the value in the diff_df creation code
    standard_time_bucketing.time_bucket_size = 1
    standard_filling.time_bucket_size = 1

    # Executes the time bucketing step and the filling step
    df = standard_time_bucketing(df=sample_dataframe_02, spark=spark)
    df = standard_filling(df=df, spark=spark)

    diff_df = create_diff_dataframe(df, standard_filling.identifier_cols_name)

    # Differences DataFrame has to be empty
    assert diff_df.count() == 0


def test_process_samples_timestamp_distance_wrong(
    spark, sample_dataframe_03, standard_time_bucketing, standard_filling
):
    """Tests that there are two cases in which the difference is different from one
    day. In the provided DataFrame the legal hour change was included. So this test
    should produce one sample differnt from 1 day duration for each user. Two users
    are present in the DataFrame so the diff_df count should be 2."""

    # Sets time_bucket_size to 1 day. If you wan to change it remember to change also
    # the value in the diff_df creation code
    standard_time_bucketing.time_bucket_size = 1
    standard_filling.time_bucket_size = 1

    # Executes the time bucketing step and the filling step
    df = standard_time_bucketing(df=sample_dataframe_03, spark=spark)
    df = standard_filling(df=df, spark=spark)

    diff_df = create_diff_dataframe(df, standard_filling.identifier_cols_name)

    # Differences DataFrame has to have two rows
    assert diff_df.count() == 2


def test_process_number_of_rows(
    spark, sample_dataframe_02, standard_time_bucketing, standard_filling
):
    """Tests that after filling the number of rows of the filled DataFrame is more or
    equal than the number of original rows."""
    standard_time_bucketing.time_bucket_size = 1
    standard_filling.time_bucket_size = 1

    df = standard_time_bucketing(df=sample_dataframe_02, spark=spark)
    df = standard_filling(df=df, spark=spark)

    assert df.count() >= sample_dataframe_02.count()
