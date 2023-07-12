from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BooleanType,
)
import pyspark.sql.functions as F
import pytest
from pydantic import ValidationError

from ts_train.step.time_bucketing import TimeBucketing


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def sample_dataframe_01(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", 5.50, "shopping", "true"),
            (348272371, "2023-01-01", 6.10, "salute", "false"),
            (348272371, "2023-01-01", 8.20, "trasporti", "false"),
            (348272371, "2023-01-01", 1.50, "trasporti", "true"),
            (348272371, "2023-01-06", 20.20, "shopping", "false"),
            (348272371, "2023-01-06", 43.00, "shopping", "true"),
            (348272371, "2023-01-06", 72.20, "shopping", "false"),
            (234984832, "2023-01-01", 15.34, "salute", "true"),
            (234984832, "2023-01-01", 36.22, "salute", "true"),
            (234984832, "2023-01-01", 78.35, "salute", "false"),
            (234984832, "2023-01-02", 2.20, "trasporti", "true"),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "DATA_TRANSAZIONE",
            "IMPORTO",
            "CA_CATEGORY_LIV0",
            "IS_CARTA",
        ],
    )

    df = df.withColumn(
        "DATA_TRANSAZIONE", F.to_timestamp(F.col("DATA_TRANSAZIONE"), "yyyy-MM-dd")
    )

    return df


@pytest.fixture
def sample_dataframe_empty(spark):
    schema = StructType(
        [
            StructField("ID_BIC_CLIENTE", IntegerType(), False),
            StructField("DATA_TRANSAZIONE", StringType(), False),
            StructField("IMPORTO", IntegerType(), False),
            StructField("CA_CATEGORY_LIV0", StringType(), False),
            StructField("IS_CARTA", BooleanType(), False),
        ]
    )
    df = spark.createDataFrame([], schema=schema)

    df = df.withColumn(
        "DATA_TRANSAZIONE", F.to_timestamp(F.col("DATA_TRANSAZIONE"), "yyyy-MM-dd")
    )

    return df


@pytest.fixture
def standard_time_bucketing():
    return TimeBucketing(
        time_column_name="DATA_TRANSAZIONE",
        time_bucket_size=2,
        time_bucket_granularity="days",
        time_bucket_col_name="bucket",
    )


def test_time_bucketing_init():
    with pytest.raises(ValidationError):
        TimeBucketing(
            time_column_name="DATA_TRANSAZIONE",
            time_bucket_size=2,
            time_bucket_granularity="typo",
            time_bucket_col_name="bucket",
        )


def test_time_bucketing_preprocess_emptiness(
    sample_dataframe_empty, standard_time_bucketing
):
    """
    Tests if _preprocess method raises ValueError("Empty DataFrame") in case you provide
    an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_time_bucketing._preprocess(sample_dataframe_empty)


@pytest.mark.parametrize(
    "time_column_name", ["CA_CATEGORY_LIV0", "not_existing_column_name"]
)
def test_time_bucketing_preprocess_time_column_not_existing(
    sample_dataframe_01, standard_time_bucketing, time_column_name
):
    """
    Tests if _preprocess method raises ValueError("Column {time_column_name} not a
    timestamp column) in the case we give the method a column which is not a time or we
    give a column name not present in the DataFrame.
    """
    standard_time_bucketing.time_column_name = time_column_name

    with pytest.raises(ValueError) as e_info:
        standard_time_bucketing._preprocess(sample_dataframe_01)

    assert (
        str(e_info.value)
        == f"Column {standard_time_bucketing.time_column_name} not a timestamp column"
        or str(e_info.value)
        == f"Column {standard_time_bucketing.time_column_name} is not a column"
    )
