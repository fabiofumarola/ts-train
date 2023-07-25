import pytest

from pydantic import ValidationError
from pyspark.sql import functions as F
from ts_train.step.filling import Filling  # type: ignore
from pyspark.sql.functions import col
from pyspark.sql.window import Window


# TESTING INIT METHOD


@pytest.mark.parametrize("time_bucket_size", [-2, "2", "-2", 2.0, 0])
def test_init_wrong_time_bucket_size(time_bucket_size):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    with pytest.raises(ValidationError):
        Filling(
            time_bucket_col_name="bucket",
            identifier_cols_name=["ID_BIC_CLIENTE"],
            time_bucket_size=time_bucket_size,
            time_bucket_granularity="days",
            new_timestamp_col_name="timestamp",
        )


def test_init_wrong_time_bucket_granularity():
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_granularity is not of an allowed value.
    """
    with pytest.raises(ValidationError):
        Filling(
            time_bucket_col_name="bucket",
            identifier_cols_name=["ID_BIC_CLIENTE"],
            time_bucket_size=2,
            time_bucket_granularity="typo",
            new_timestamp_col_name="timestamp",
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
    "identifier_cols_name", [["typo"], ["typo1", "typo2"], ["typo", "bucket"]]
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
    standard_filling = Filling(
        time_bucket_col_name="bucket",
        identifier_cols_name=identifier_cols_name,
        time_bucket_size=1,
        time_bucket_granularity="days",  # type: ignore
        new_timestamp_col_name="timestamp",
    )

    df = standard_time_bucketing(sample_dataframe_02, spark)

    with pytest.raises(
        ValueError,
        match=r"Column (" + identifier_cols_name[0] + r") is not a column",
    ):
        standard_filling._preprocess(df)


# TESTING _PROCESS METHOD

"""
@pytest.mark.parametrize("time_bucket_granularity", ["days", "hours"])
@pytest.mark.parametrize("time_bucket_size", [1, 3, 10, 20, 100])
def test_process_samples_timestamp_with_unix_timestamp(
    spark, sample_dataframe_pre_filling, time_bucket_size, time_bucket_granularity
):
    time_column_name = "timestamp"
    identifier_cols_name = ["ID_BIC_CLIENTE"]

    # Tests that the difference between each sample for each id is one day.
    standard_filling = Filling(
        time_bucket_col_name="bucket",
        identifier_cols_name=identifier_cols_name,
        time_bucket_size=time_bucket_size,
        time_bucket_granularity=time_bucket_granularity,
        new_timestamp_col_name=time_column_name,
    )

    # Executes the time bucketing step and the filling step
    df_after_filling = standard_filling(df=sample_dataframe_pre_filling, spark=spark)
    # df_after_filling.show()

    # Check if the specified column contains any null values
    contains_nulls = df_after_filling.where(col(time_column_name).isNull()).count() > 0
    assert not contains_nulls

    # Convert the 'timestamp' column to Unix timestamp
    df_after_filling = df_after_filling.withColumn(
        "timestamp_unix", F.unix_timestamp(time_column_name)
    )

    # Group by 'ID_BIC_CLIENTE' and collect the list of all 'timestamp_unix' values
    # for each user
    timestamps_per_user = df_after_filling.groupBy(*identifier_cols_name).agg(
        F.collect_list("timestamp_unix").alias("timestamps_list")
    )

    all_users = timestamps_per_user.select(*identifier_cols_name).distinct().collect()
    for user_row in all_users:
        user_identifier_values = user_row.asDict()
        # Build a single filter condition for all identifier columns
        filter_condition = (
            col(col_name) == lit(col_value)
            for col_name, col_value in user_identifier_values.items()
        )
        user_timestamps = (
            timestamps_per_user.filter(reduce(lambda x, y: x & y, filter_condition))
            .select("timestamps_list")
            .collect()[0][0]
        )
        # Calculate the differences between each element and the next one
        # using list comprehension
        differences = [
            user_timestamps[i + 1] - user_timestamps[i]
            for i in range(len(user_timestamps) - 1)
        ]

        # Check if all differences are equal
        assert all(difference == differences[0] for difference in differences)
"""


@pytest.mark.parametrize("time_bucket_granularity", ["days", "hours", "minutes"])
@pytest.mark.parametrize("time_bucket_size", [1, 3, 10, 20, 100])
def test_process_samples_timestamp_distance_with_spark_utility(
    spark, sample_dataframe_pre_filling, time_bucket_size, time_bucket_granularity
):
    time_column_name = "timestamp"
    identifier_cols_name = ["ID_BIC_CLIENTE"]

    # Tests that the difference between each sample for each id is one day.
    standard_filling = Filling(
        time_bucket_col_name="bucket",
        identifier_cols_name=identifier_cols_name,
        time_bucket_size=time_bucket_size,
        time_bucket_granularity=time_bucket_granularity,
        new_timestamp_col_name=time_column_name,
    )

    df_after_filling = standard_filling(df=sample_dataframe_pre_filling, spark=spark)

    # Check if the specified column contains any null values
    contains_nulls = df_after_filling.where(col(time_column_name).isNull()).count() > 0
    assert not contains_nulls

    # Create a Window specification with partitioning by 'ID_BIC_CLIENTE' and ordering
    # by 'timestamp'
    window_spec = Window.partitionBy(identifier_cols_name).orderBy(time_column_name)

    # Calculate the time differences between all timestamps
    df_after_filling = df_after_filling.withColumn(
        f"shifted_{time_column_name}", F.lag(time_column_name, 1).over(window_spec)
    )
    df_after_filling = df_after_filling.withColumn(
        "time_diff",
        F.col(time_column_name).cast("long")
        - F.col(f"shifted_{time_column_name}").cast("long"),
    )

    # check for the first timestamp row for each user
    # count the number of id_columns unique values
    num_unique_ID_BIC_CLIENTE = (
        df_after_filling.select(identifier_cols_name).distinct().count()
    )
    # count the number of null values in the time_diff column
    num_null_time_diff = df_after_filling.filter(col("time_diff").isNull()).count()
    assert num_unique_ID_BIC_CLIENTE == num_null_time_diff

    if df_after_filling.count() > num_null_time_diff:
        difference_between_timestamps = len(
            df_after_filling.select(F.collect_set("time_diff")).collect()[0][0]
        )
        assert difference_between_timestamps == 1

    df_after_filling.show(truncate=False)
