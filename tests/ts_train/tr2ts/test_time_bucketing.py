import pytest
from pydantic import ValidationError
from pyspark.sql import functions as F
from ts_train.tr2ts.time_bucketing import TimeBucketing  # type: ignore
from pyspark.sql.functions import col, lit
from functools import reduce
from pyspark.sql.types import StringType


# TESTING INIT METHOD
@pytest.mark.parametrize("time_bucket_size", [-2, "2", "-2", 2.0, 0])
def test_init_wrong_time_bucket_size(time_bucket_size):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    with pytest.raises(ValidationError):
        #### APPLICO IL TIME BUCKETING
        TimeBucketing(
            time_col_name="DATA_TRANSAZIONE",
            time_bucket_size=time_bucket_size,
            time_bucket_granularity="day",
        )


# TESTING _PREPROCESS METHOD


def test_incorrect_granularity():
    """
    Tests if _preprocess method raises ValidationError in case you provide an not legit
    granularity.
    """
    with pytest.raises(ValidationError):
        TimeBucketing(
            time_col_name="DATA_TRANSAZIONE",
            time_bucket_size=1,
            time_bucket_granularity="x",  # type: ignore
        )


def test_preprocess_emptiness(sample_dataframe_empty, standard_time_bucketing):
    """
    Tests if _preprocess method raises ValueError("Empty DataFrame") in case you provide
    an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_time_bucketing._preprocess(sample_dataframe_empty)


def test_preprocess_time_column_not_existing(
    sample_dataframe_04, standard_time_bucketing
):
    """
    Tests if _preprocess method raises ValueError("Column {time_col_name} is not a
    column) in the case we give a column name not present in the DataFrame.
    """
    standard_time_bucketing.time_col_name = "not_existing_column_name"

    with pytest.raises(
        ValueError,
        match=f"Column {standard_time_bucketing.time_col_name} is not present",
    ):
        standard_time_bucketing._preprocess(sample_dataframe_04)


def test_preprocess_time_column_not_timestamp(
    sample_dataframe_04, standard_time_bucketing
):
    """Tests if _preprocess method raises ValueError("Column {time_col_name} not a
    timestamp column) in the case we give the method a column which is not a time.
    """
    standard_time_bucketing.time_col_name = "IMPORTO"

    with pytest.raises(
        ValueError,
        match=(
            f"Column {standard_time_bucketing.time_col_name} is not a timestamp column"
        ),
    ):
        standard_time_bucketing._preprocess(sample_dataframe_04)


# Testing _PROCESS METHOD
def test_process_new_column_added(sample_dataframe_04, spark, standard_time_bucketing):
    """Tests if _process method adds a new column."""
    result_df = standard_time_bucketing._process(sample_dataframe_04, spark)

    assert len(sample_dataframe_04.columns) + 2 == len(result_df.columns), (
        f"number of old columns {len(sample_dataframe_04.columns)}. Number of new"
        f" dataframe columns {len(result_df.columns)}"
    )


def test_process_new_column_type(
    sample_dataframe_04, spark, standard_time_bucketing: TimeBucketing
):
    """Tests if _process method adds a new column with the right datactype."""
    result_df = standard_time_bucketing._process(sample_dataframe_04, spark)

    # assert str(result_df.schema["bucket_start"].dataType) == "StringType()"
    assert str(result_df.schema["bucket_start"].dataType) == "TimestampType()"


@pytest.mark.slow
@pytest.mark.parametrize("time_bucket_granularity", ["h", "d", "w", "m", "y"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 3, 8, 12])
def test_create_timeline(
    sample_dataframe_04,
    standard_time_bucketing,
    time_bucket_granularity,
    time_bucket_size,
):
    """
    Tests if _create_timeline method creates a timeline from min_date to max_date.
    Specificly it certificate that the dates are not changed during the generation
    of timeline
    """
    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    # assicurarsi che il min_date corrisponde all' inizio della timeline
    assert (
        min_date == timeline[0]
    ), f"min_date: {min_date} != date_range[0]: {timeline[0]}"

    # se l' intevallo temporale non è divisibile per lo step_size, il range si ferma
    # all' ultimo step possibile senza superare l' intervallo superiore.
    assert (
        max_date >= timeline[-1]
    ), f"max_date: {max_date} != date_range[-1]: {timeline[-1]}"


@pytest.mark.slow
@pytest.mark.parametrize("time_col_name", ["bucket_start", "bucket_end"])
@pytest.mark.parametrize("time_bucket_granularity", ["h", "d", "w", "m", "y"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 3, 8, 12])
def test_buckets_monotonicity(
    sample_dataframe_04,
    standard_time_bucketing,
    time_bucket_granularity,
    time_bucket_size,
    time_col_name,
    spark,
):
    """
    Tests if _create_df_with_buckets method creates a dataframe with buckets
    and that all bucket are ordered by time, so every bucket start is greater the
    the previous one
    """

    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    bucket_df = standard_time_bucketing._create_df_with_buckets(spark, timeline)

    all_dates = bucket_df.select(time_col_name).collect()

    for id_date in range(1, len(all_dates)):
        assert all_dates[id_date] > all_dates[id_date - 1], (
            f"during test for {time_col_name}, at row"
            f" {id_date}, the date {all_dates[id_date]} is <= then date at"
            f" {id_date} that is {all_dates[id_date-1]}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("time_bucket_granularity", ["h", "d", "w", "m", "y"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 3, 8, 12])
# Verifica che non ci sia stata corruzione di dai durante la conversione della timeline
# da pandas a spark
def test_time_range(
    sample_dataframe_04,
    standard_time_bucketing,
    time_bucket_granularity,
    time_bucket_size,
    spark,
):
    """
    Tests if _create_df_with_buckets method turn the timeline into a spark dataframe
    with bucket without changing date and mantain the firts and last date
    """

    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    bucket_df = standard_time_bucketing._create_df_with_buckets(spark, timeline)

    # assicurarsi che non ci sia stata corruzione di dai durante la conversione da
    # timeline a spark df
    min_date = str(min_date)
    max_date = str(max_date)
    first_element = bucket_df.first()["bucket_start"]
    last_element = bucket_df.tail(1)[0]["bucket_end"]

    assert first_element == str(
        min_date
    ), f"first_element: {first_element} min_date: {min_date}"

    # since last_element also contains the offset, it should be bigger then the max_date
    assert last_element > str(
        max_date
    ), f"last_element: {last_element} > max_date: {max_date}"


@pytest.mark.slow
@pytest.mark.parametrize("time_col_name", ["bucket_start", "bucket_end"])
@pytest.mark.parametrize("time_bucket_granularity", ["h", "d", "w", "m", "y"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 3, 8, 12])
# assicurarsi che ogni bucket sia equidistante
def test_bucket_df_null_values(
    sample_dataframe_04,
    standard_time_bucketing,
    time_bucket_granularity,
    time_bucket_size,
    time_col_name,
    spark,
):
    """
    Test if _create_df_with_buckets method creates a dataframe with buckets and that
    there are no Null rows
    """
    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    bucket_df = standard_time_bucketing._create_df_with_buckets(spark, timeline)

    # Check if the specified column contains any null values
    contains_nulls = bucket_df.where(col(time_col_name).isNull()).count() > 0
    assert not contains_nulls, f"Column '{time_col_name}' contains null values."


@pytest.mark.parametrize("time_col_name", ["bucket_start", "bucket_end"])
@pytest.mark.parametrize("time_bucket_granularity", ["m"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 9])
def test_all_buckets_are_equidistant_multi_user(
    sample_dataframe_04,
    standard_time_bucketing,
    time_bucket_granularity,
    time_bucket_size,
    time_col_name,
    spark,
):
    """
    Test if all buckets are equidistant for all users. So for each user, calculate the
    distance between the beginning of each bucket and the beginning of the next one. At
    the end it checks that the calculated distances are always the same.
    """
    identifier_cols_name = [standard_time_bucketing.time_col_name] + ["ID_BIC_CLIENTE"]

    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    bucket_df = standard_time_bucketing._create_df_with_buckets(spark, timeline)
    final_df = standard_time_bucketing._bucketize_data(bucket_df, sample_dataframe_04)

    final_df = final_df.withColumn("timestamp_unix", F.unix_timestamp(time_col_name))
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


@pytest.mark.slow
@pytest.mark.parametrize("time_bucket_granularity", ["d", "w", "m", "y"])
@pytest.mark.parametrize("time_bucket_size", [1, 2, 3, 8, 12])
def test_all_transactions_are_in_the_correct_bucket(
    sample_dataframe_04,
    standard_time_bucketing,
    time_bucket_granularity,
    time_bucket_size,
    spark,
):
    """
    This test checks if all transactions are in the correct bucket. So the date is
    always in between the bucket_start and bucket_end.
    """
    standard_time_bucketing.time_bucket_granularity = time_bucket_granularity
    standard_time_bucketing.time_bucket_size = time_bucket_size

    timeline, min_date, max_date = standard_time_bucketing._create_timeline(
        sample_dataframe_04
    )

    bucket_df = standard_time_bucketing._create_df_with_buckets(spark, timeline)
    final_df = standard_time_bucketing._bucketize_data(bucket_df, sample_dataframe_04)

    final_df = final_df.withColumn(
        standard_time_bucketing.time_col_name,
        col(standard_time_bucketing.time_col_name).cast(StringType()),
    )
    final_df = final_df.withColumn(
        "bucket_start",
        col("bucket_start").cast(StringType()),
    )
    final_df = final_df.withColumn(
        "bucket_end",
        col("bucket_end").cast(StringType()),
    )

    all_dates = final_df.select(standard_time_bucketing.time_col_name).collect()
    all_bucket_starts = final_df.select("bucket_start").collect()
    all_bucket_ends = final_df.select("bucket_end").collect()

    for i in range(len(all_dates)):
        assert all_bucket_starts[i] <= all_dates[i] < all_bucket_ends[i], (
            f"{all_bucket_starts[i]} <= {all_dates[i]} < {all_bucket_ends[i]} is False"
            f" for i = {i}"
        )
