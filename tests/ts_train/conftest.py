from typing import *

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BooleanType,
)
import pytest

from ts_train.step.time_bucketing import TimeBucketing  # type: ignore
from ts_train.step.filling import Filling  # type: ignore
from ts_train.step.aggregation import Aggregation  # type: ignore
from ts_train.common.utils import (  # type: ignore
    cast_column_to_timestamp,  # type: ignore
    create_timestamps_struct,  # type: ignore
)


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.getOrCreate()  # type: ignore


"""
# Fixture to create a SparkSession
@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("pytest_spark_tests") \
        .getOrCreate()
    yield spark
    spark.stop()
"""


@pytest.fixture
def sample_dataframe_01(spark):
    df = spark.createDataFrame(
        data=[
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

    return cast_column_to_timestamp(df=df, col_name="DATA_TRANSAZIONE")


@pytest.fixture
def sample_dataframe_01_bucketed(spark):
    df = spark.createDataFrame(
        data=[
            (
                348272371,
                "2023-01-01",
                55,
                "shopping",
                True,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                348272371,
                "2023-01-01",
                61,
                "salute",
                False,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                348272371,
                "2023-01-01",
                82,
                "trasporti",
                False,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                348272371,
                "2023-01-01",
                15,
                "trasporti",
                True,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                348272371,
                "2023-01-06",
                202,
                "shopping",
                False,
                "2023-01-06",
                "2023-01-07",
            ),
            (
                348272371,
                "2023-01-06",
                430,
                "shopping",
                True,
                "2023-01-06",
                "2023-01-07",
            ),
            (
                348272371,
                "2023-01-06",
                722,
                "shopping",
                False,
                "2023-01-06",
                "2023-01-07",
            ),
            (
                234984832,
                "2023-01-01",
                153,
                "salute",
                True,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                234984832,
                "2023-01-01",
                362,
                "salute",
                True,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                234984832,
                "2023-01-01",
                783,
                "salute",
                False,
                "2023-01-01",
                "2023-01-02",
            ),
            (
                234984832,
                "2023-01-02",
                22,
                "trasporti",
                True,
                "2023-01-02",
                "2023-01-03",
            ),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "DATA_TRANSAZIONE",
            "IMPORTO",
            "CA_CATEGORY_LIV0",
            "IS_CARTA",
            "bucket_start",
            "bucket_end",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


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

    return cast_column_to_timestamp(df=df, col_name="DATA_TRANSAZIONE")


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

    return cast_column_to_timestamp(df=df, col_name="DATA_TRANSAZIONE")


@pytest.fixture
def sample_dataframe_pre_filling(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 61, 55, 97),
            (348272371, "2023-01-06", "2023-01-07", None, 1354, None),
            (234984832, "2023-01-01", "2023-01-02", 1298, None, None),
            (234984832, "2023-01-02", "2023-01-03", None, None, 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "salute",
            "shopping",
            "trasporti",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


@pytest.fixture
def sample_dataframe_empty(spark):
    df = spark.createDataFrame(
        data=[],
        schema=StructType(
            [
                StructField("ID_BIC_CLIENTE", IntegerType(), False),
                StructField("DATA_TRANSAZIONE", StringType(), False),
                StructField("IMPORTO", IntegerType(), False),
                StructField("CA_CATEGORY_LIV0", StringType(), False),
                StructField("IS_CARTA", BooleanType(), False),
            ]
        ),
    )

    return cast_column_to_timestamp(df=df, col_name="DATA_TRANSAZIONE")


@pytest.fixture
def standard_time_bucketing():
    return TimeBucketing(
        time_zone="Europe/Rome",
        time_column_name="DATA_TRANSAZIONE",
        time_bucket_size=2,
        time_bucket_granularity="days",  # type: ignore
        time_bucket_col_name="bucket",
    )


@pytest.fixture
def standard_aggregation():
    return Aggregation(
        numerical_col_name=["IMPORTO"],
        identifier_cols_name=["ID_BIC_CLIENTE"],
        all_aggregation_filters=[],
        agg_funcs=["sum"],
    )


@pytest.fixture
def standard_filling():
    return Filling(
        time_bucket_col_name="bucket",
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_size=2,
        time_bucket_granularity="days",  # type: ignore
        new_timestamp_col_name="timestamp",
    )


# Expected results from aggregation steps

# _pivot method


# filter=("CA_CATEGORY_LIV0", [])
@pytest.fixture
def result_aggregation_pivot_string_cat_no_options(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 61, 55, 97),
            (348272371, "2023-01-06", "2023-01-07", None, 1354, None),
            (234984832, "2023-01-01", "2023-01-02", 1298, None, None),
            (234984832, "2023-01-02", "2023-01-03", None, None, 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_IMPORTO_by_CA_CATEGORY_LIV0_(salute)",
            "sum_IMPORTO_by_CA_CATEGORY_LIV0_(shopping)",
            "sum_IMPORTO_by_CA_CATEGORY_LIV0_(trasporti)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# filter=("CA_CATEGORY_LIV0", ["salute"])
@pytest.fixture
def result_aggregation_pivot_string_cat_one_option(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 61),
            (348272371, "2023-01-06", "2023-01-07", None),
            (234984832, "2023-01-01", "2023-01-02", 1298),
            (234984832, "2023-01-02", "2023-01-03", None),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_IMPORTO_by_CA_CATEGORY_LIV0_(salute)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# filter=("CA_CATEGORY_LIV0", ["salute", "trasporti"])
@pytest.fixture
def result_aggregation_pivot_string_cat_two_options(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 61, 97),
            (348272371, "2023-01-06", "2023-01-07", None, None),
            (234984832, "2023-01-01", "2023-01-02", 1298, None),
            (234984832, "2023-01-02", "2023-01-03", None, 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_IMPORTO_by_CA_CATEGORY_LIV0_(salute)",
            "sum_IMPORTO_by_CA_CATEGORY_LIV0_(trasporti)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# filter=("IS_CARTA", [])
@pytest.fixture
def result_aggregation_pivot_bool_cat_no_options(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 70, 143),
            (348272371, "2023-01-06", "2023-01-07", 430, 924),
            (234984832, "2023-01-01", "2023-01-02", 515, 783),
            (234984832, "2023-01-02", "2023-01-03", 22, None),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_IMPORTO_by_IS_CARTA_(True)",
            "sum_IMPORTO_by_IS_CARTA_(False)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# _selecting method


# filters=[("CA_CATEGORY_LIV0", ["salute", "trasporti"]), ("IS_CARTA", ["true"])]
@pytest.fixture
def result_aggregation_select_salute_trasporti_is_carta(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 15),
            (348272371, "2023-01-06", "2023-01-07", None),
            (234984832, "2023-01-01", "2023-01-02", 515),
            (234984832, "2023-01-02", "2023-01-03", 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_of_IMPORTO_by_CA_CATEGORY_LIV0_(salute_trasporti)_and_by_IS_CARTA_(True)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# filters=[("CA_CATEGORY_LIV0", ["salute"]), ("IS_CARTA", ["true"])]
@pytest.fixture
def result_aggregation_select_salute_is_carta(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", None),
            (348272371, "2023-01-06", "2023-01-07", None),
            (234984832, "2023-01-01", "2023-01-02", 515),
            (234984832, "2023-01-02", "2023-01-03", None),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_of_IMPORTO_by_CA_CATEGORY_LIV0_(salute)_and_by_IS_CARTA_(True)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# filters=[("CA_CATEGORY_LIV0", ["salute", "trasporti"])]
@pytest.fixture
def result_aggregation_select_salute_trasporti(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 158),
            (348272371, "2023-01-06", "2023-01-07", None),
            (234984832, "2023-01-01", "2023-01-02", 1298),
            (234984832, "2023-01-02", "2023-01-03", 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum_of_IMPORTO_by_CA_CATEGORY_LIV0_(salute_trasporti)",
        ],
    )

    return create_timestamps_struct(
        df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
    )


# _process method


# filters=[
#    [("CA_CATEGORY_LIV0", ["salute", "trasporti"])],
#    [("CA_CATEGORY_LIV0", ["salute"])],
# ]
# @pytest.fixture
# def result_aggregation_process_01(spark):
#     df = spark.createDataFrame(
#         [
#             (348272371, "2023-01-01", "2023-01-02", 158, 61),
#             (348272371, "2023-01-06", "2023-01-07", None, None),
#             (234984832, "2023-01-01", "2023-01-02", 1298, 1298),
#             (234984832, "2023-01-02", "2023-01-03", 22, None),
#         ],
#         schema=[
#             "ID_BIC_CLIENTE",
#             "bucket_start",
#             "bucket_end",
#             "sum_of_IMPORTO_by_CA_CATEGORY_LIV0_(salute_trasporti)",
#             "sum_of_IMPORTO_by_CA_CATEGORY_LIV0_(salute_trasporti)",
#         ],
#     )

#     return create_timestamps_struct(
#         df=df, cols_name=("bucket_start", "bucket_end"), struct_col_name="bucket"
#     )
