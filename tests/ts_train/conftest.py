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

from ts_train.step.time_bucketing import TimeBucketing  # type: ignore
from ts_train.step.filling import Filling  # type: ignore


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
def sample_dataframe_02(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01"),
            (348272371, "2023-01-02"),
            (348272371, "2023-01-06"),
            (234984832, "2023-01-01"),
            (234984832, "2023-01-02"),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "DATA_TRANSAZIONE",
        ],
    )

    df = df.withColumn(
        "DATA_TRANSAZIONE", F.to_timestamp(F.col("DATA_TRANSAZIONE"), "yyyy-MM-dd")
    )

    return df


@pytest.fixture
def sample_dataframe_03(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-03-25"),
            (348272371, "2023-03-26"),
            (348272371, "2023-03-28"),
            (234984832, "2023-03-26"),
            (234984832, "2023-03-27"),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "DATA_TRANSAZIONE",
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
        time_zone="Europe/Rome",
        time_column_name="DATA_TRANSAZIONE",
        time_bucket_size=2,
        time_bucket_granularity="days",
        time_bucket_col_name="bucket",
    )


@pytest.fixture
def standard_filling():
    return Filling(
        time_bucket_col_name="bucket",
        identifier_cols_name="ID_BIC_CLIENTE",
        time_bucket_size=2,
        time_bucket_granularity="days",
    )
