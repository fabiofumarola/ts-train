from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BooleanType,
)
import pytest

from ts_train.tr2ts.time_bucketing import TimeBucketing  # type: ignore
from ts_train.tr2ts.filling import Filling  # type: ignore

from ts_train.common.utils import (  # type: ignore
    cast_column_to_timestamp,  # type: ignore
    cast_columns_to_timestamp,  # type: ignore
)

from ts_train.tr2ts.aggregating import (  # type: ignore
    Aggregating,  # type: ignore
    Aggregation,  # type: ignore
    Filter,  # type: ignore
    Pivot,  # type: ignore
)  # type: ignore


@pytest.fixture(scope="session")
def spark():
    spark_session = SparkSession.builder.getOrCreate()  # type: ignore
    return spark_session


@pytest.fixture
def sample_dataframe_01(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, "2023-01-01", 5.50, "shopping", True),
            (348272371, "2023-01-01", 6.10, "salute", False),
            (348272371, "2023-01-01", 8.20, "trasporti", False),
            (348272371, "2023-01-01", 1.50, "trasporti", True),
            (348272371, "2023-01-06", 20.20, "shopping", False),
            (348272371, "2023-01-06", 43.00, "shopping", True),
            (348272371, "2023-01-06", 72.20, "shopping", False),
            (234984832, "2023-01-01", 15.34, "salute", True),
            (234984832, "2023-01-01", 36.22, "salute", True),
            (234984832, "2023-01-01", 78.35, "salute", False),
            (234984832, "2023-01-02", 2.20, "trasporti", True),
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

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


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
def sample_dataframe_04(spark):
    # Definisci la struttura dello schema
    schema = [
        "ID_BIC_CLIENTE",
        "DATA_TRANSAZIONE",
        "IMPORTO",
    ]

    # Dati di esempio
    data = [
        (1, "2023-04-16 01:00:00", 30.0),
        (1, "2023-04-26 02:00:00", 50.0),
        (1, "2023-04-26 05:00:00", 40.0),
        (1, "2023-04-27 02:00:00", 70.0),
        (1, "2023-04-29 04:00:00", 40.0),
        (2, "2023-04-26 03:00:00", 40.0),
        (2, "2023-04-26 07:00:00", 50.0),
        (2, "2023-04-27 02:00:00", 60.0),
        (2, "2023-04-27 03:00:00", 20.0),
        (2, "2023-04-29 02:00:00", 23.0),
        (3, "2023-04-25 02:00:00", 40.0),
        (3, "2023-04-26 04:00:00", 23.0),
        (3, "2023-04-27 01:00:00", 60.0),
        (3, "2023-04-27 04:00:00", 30.0),
        (3, "2023-04-28 02:00:00", 70.0),
        (3, "2023-05-17 00:00:00", 12.0),
        (3, "2024-03-26 02:00:00", 20.0),
    ]

    # Crea il dataframe
    df = spark.createDataFrame(data, schema=schema)
    df = cast_column_to_timestamp(
        df=df, col_name="DATA_TRANSAZIONE", format="yyyy-MM-dd HH:mm:ss"
    )
    return df


@pytest.fixture
def sample_dataframe_for_aggregation_process_bucketed(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, "2023-01-01", "2023-01-02", 5, "shopping", "carta", True),
            (348272371, "2023-01-01", "2023-01-02", 6, "salute", "cash", False),
            (348272371, "2023-01-01", "2023-01-02", 8, "trasporti", "cash", False),
            (348272371, "2023-01-01", "2023-01-02", 1, "trasporti", "carta", True),
            (348272371, "2023-01-06", "2023-01-07", 20, "shopping", "bitcoin", False),
            (348272371, "2023-01-06", "2023-01-07", 43, "shopping", "carta", True),
            (348272371, "2023-01-06", "2023-01-07", 72, "shopping", "cash", False),
            (234984832, "2023-01-01", "2023-01-02", 15, "salute", "carta", True),
            (234984832, "2023-01-01", "2023-01-02", 36, "salute", "carta", True),
            (234984832, "2023-01-01", "2023-01-02", 78, "salute", "cash", False),
            (234984832, "2023-01-02", "2023-01-03", 2, "trasporti", "carta", True),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "IMPORTO",
            "CA_CATEGORY_LIV0",
            "METODO_PAGAMENTO",
            "IS_CARTA",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


@pytest.fixture
def standard_time_bucketing():
    return TimeBucketing(
        time_col_name="DATA_TRANSAZIONE",
        time_bucket_size=1,
        time_bucket_granularity="day",
    )


@pytest.fixture
def standard_aggregator():
    return Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                filters=[Filter(col_name="IS_CARTA", operator="==", value=True)],
            )
        ],
    )


@pytest.fixture
def complex_aggregator():
    return Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                pivot=Pivot("METODO_PAGAMENTO", "in", None),
            ),
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                pivot=Pivot("CA_CATEGORY_LIV0", "in", ["shopping", "salute"]),
            ),
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                filters=[Filter("CA_CATEGORY_LIV0", "in", ["shopping", "salute"])],
            ),
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                filters=[
                    Filter("IS_CARTA", "=", False),
                    Filter("METODO_PAGAMENTO", "=", "carta"),
                ],
            ),
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                filters=[
                    Filter("IS_CARTA", "=", True),
                    Filter("METODO_PAGAMENTO", "in", ["cash", "bitcoin"]),
                ],
            ),
        ],
    )


@pytest.fixture
def standard_filling(standard_time_bucketing):
    return Filling(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_step=standard_time_bucketing,
    )


@pytest.fixture
def no_pivot_no_filter_expected_df(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 213),
            (348272371, "2023-01-06", "2023-01-07", 1354),
            (234984832, "2023-01-01", "2023-01-02", 1298),
            (234984832, "2023-01-02", "2023-01-03", 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum(IMPORTO)",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


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
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=salute",
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=shopping",
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=trasporti",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# pivot=Pivot("CA_CATEGORY_LIV0", "in", ["salute"])
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
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=salute",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# pivot=Pivot("CA_CATEGORY_LIV0", "in", ["salute", "trasporti"])
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
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=salute",
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=trasporti",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# pitov=Pivot("IS_CARTA", "in", [])
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
            "sum(IMPORTO)_where_IS_CARTA=True",
            "sum(IMPORTO)_where_IS_CARTA=False",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# pitov=Pivot("IS_CARTA", "in", [True])
@pytest.fixture
def result_aggregation_pivot_bool_cat_true(spark):
    df = spark.createDataFrame(
        [
            (348272371, "2023-01-01", "2023-01-02", 70),
            (348272371, "2023-01-06", "2023-01-07", 430),
            (234984832, "2023-01-01", "2023-01-02", 515),
            (234984832, "2023-01-02", "2023-01-03", 22),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum(IMPORTO)_where_IS_CARTA=True",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# filter


# filters=[Filter("CA_CATEGORY_LIV0", "in", ["salute", "trasporti"]),
# Filter("IS_CARTA", "=", True)]
@pytest.fixture
def result_aggregator_filters_salute_trasporti_is_carta(spark):
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
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0[salute_trasporti]&IS_CARTA=True",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# filters=[Filter("CA_CATEGORY_LIV0", "=", "salute"]), Filter("IS_CARTA", "=", "True])]
@pytest.fixture
def result_aggregator_filters_salute_is_carta(spark):
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
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=salute&IS_CARTA=True",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# filters=[Filter("CA_CATEGORY_LIV0", "in", ["salute", "trasporti"])]
@pytest.fixture
def result_aggregator_filters_salute_trasporti(spark):
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
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0[salute_trasporti]",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df


# _process method


@pytest.fixture
def expected_aggregated_df_01(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, "2023-01-06", "2023-01-07", 43, 72, 20, 135, 0, 135, 0, 0),
            (234984832, "2023-01-01", "2023-01-02", 51, 78, 0, 0, 129, 129, 0, 0),
            (234984832, "2023-01-02", "2023-01-03", 2, 0, 0, 0, 0, 0, 0, 0),
            (348272371, "2023-01-01", "2023-01-02", 6, 14, 0, 5, 6, 11, 0, 0),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "bucket_end",
            "sum(IMPORTO)_where_METODO_PAGAMENTO=carta",
            "sum(IMPORTO)_where_METODO_PAGAMENTO=cash",
            "sum(IMPORTO)_where_METODO_PAGAMENTO=bitcoin",
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=shopping",
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0=salute",
            "sum(IMPORTO)_where_CA_CATEGORY_LIV0[shopping_salute]",
            "sum(IMPORTO)_where_IS_CARTA=False&METODO_PAGAMENTO=carta",
            "sum(IMPORTO)_where_IS_CARTA=True&METODO_PAGAMENTO[cash_bitcoin]",
        ],
    )

    df = cast_columns_to_timestamp(df, cols_name=["bucket_start", "bucket_end"])

    return df
