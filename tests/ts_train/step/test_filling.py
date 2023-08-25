import pytest
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from functools import reduce

# from ts_train.step.time_bucketing import TimeBucketing  # type: ignore
from ts_train.step.filling import Filling  # type: ignore


# TESTING _PREPROCESS METHOD
def test_preprocess_emptiness(sample_dataframe_empty, standard_filling):
    """Tests if _preprocess method raises ValueError("Empty DataFrame") in case you
    provide an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_filling._preprocess(sample_dataframe_empty)


@pytest.mark.parametrize(
    "identifier_cols_name", [["typo"], ["typo1", "typo2"], ["typo", "bucket"]]
)
def test_preprocess_identifier_cols_name_not_existing(
    spark,
    sample_dataframe_01,
    standard_time_bucketing,
    standard_filling,
    identifier_cols_name,
):
    """Tests if _preprocess method raises ValueError("Column {column name} is not a
    column") in the case in which identifier_cols_name is a wrong column name or it is
    a list with inside a wrong column name."""
    standard_filling = Filling(
        identifier_cols_name=identifier_cols_name,
        time_bucket_step=standard_time_bucketing,
    )

    df = standard_time_bucketing(sample_dataframe_01, spark)

    with pytest.raises(
        ValueError,
        match=r"Column (" + identifier_cols_name[0] + r") is not a column",
    ):
        standard_filling._preprocess(df)


# TESTING _PROCESS METHOD

'''
@pytest.mark.parametrize("time_column_name", ["bucket_start", "bucket_end"])
@pytest.mark.parametrize("time_bucket_granularity", ["m"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 9])
def test_all_buckets_are_equidistant_multi_user(
    sample_dataframe_04,
    standard_time_bucketing,
    standard_filling,
    time_bucket_granularity,
    time_bucket_size,
    time_column_name,
    spark,
):
    """
    Test if all buckets are equidistant for all users. So for each user, calculate the
    distance between the beginning of each bucket and the beginning of the next one. At
    the end it checks that the calculated distances are always the same.
    """
    identifier_cols_name = [standard_time_bucketing.time_column_name] + [
        "ID_BIC_CLIENTE"
    ]

    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    bucket_df = standard_time_bucketing._create_df_with_buckets(spark, timeline)
    final_df = standard_time_bucketing._bucketize_data(bucket_df, sample_dataframe_04)

    final_df = final_df.withColumn("timestamp_unix", F.unix_timestamp(time_column_name))
    timestamps_per_user = final_df.groupBy(*identifier_cols_name).agg(
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
'''


def test_process_samples_timestamp_with_unix_timestamp(
    spark,
    sample_dataframe_01_bucketed,
    standard_filling,
):
    identifier_cols_name = ["ID_BIC_CLIENTE"]
    sample_dataframe_01_bucketed.show(truncate=False)

    # Executes the time bucketing step and the filling step
    df_after_filling = standard_filling(df=sample_dataframe_01_bucketed, spark=spark)
    df_after_filling.show()

    # Check if the specified column contains any null values
    contains_nulls = df_after_filling.where(col("bucket_start").isNull()).count() > 0
    assert not contains_nulls, "Column bucket_start contains null values."

    # Convert the 'timestamp' column to Unix timestamp
    df_after_filling = df_after_filling.withColumn(
        "timestamp_unix", F.unix_timestamp("bucket_start")
    )

    df_after_filling.show()
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
