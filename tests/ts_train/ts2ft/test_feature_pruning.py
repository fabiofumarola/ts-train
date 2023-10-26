import pytest
import pandas as pd

from pyspark_assert import assert_frame_equal  # type: ignore

from ts_train.ts2ft.feature_pruning import FeaturePruning  # type: ignore


@pytest.fixture
def feature_pruning():
    return FeaturePruning(
        identifier_col_name="ID_CLIENTE_BIC", target_col_name="TARGET"
    )


@pytest.fixture
def features_spark_df(spark):
    df = spark.createDataFrame(
        data=[
            (1, 1, 33, 11),
            (2, 10, 56, 22),
            (3, 100, 12, 33),
        ],
        schema=["ID_CLIENTE_BIC", "ft1", "ft2", "TARGET"],
    )

    return df


@pytest.fixture
def features_spark_df_with_null(spark):
    df = spark.createDataFrame(
        data=[
            (1, 1, 33, 11),
            (2, None, 56, 22),
            (3, 100, 12, 33),
        ],
        schema=["ID_CLIENTE_BIC", "ft1", "ft2", "TARGET"],
    )

    return df


@pytest.fixture
def features_spark_df_2(spark):
    targets_2 = [num + 3.2 for num in range(0, 1000)]

    pandas_df = pd.DataFrame(
        {
            "ID_CLIENTE_BIC": range(0, 1000),
            "ft1": [num * 2 for num in targets_2],
            "ft2": [num % 2 for num in range(0, 1000)],
            "ft3": [num % 3 for num in range(0, 1000)],
            "TARGET": targets_2,
        }
    )

    return spark.createDataFrame(pandas_df)


def test_preprocess_check_null_in_columns(
    spark, feature_pruning, features_spark_df_with_null
):
    with pytest.raises(
        ValueError, match="Columns \['ft1'\] of DataFrame must not contain NaN values"
    ):
        feature_pruning(df=features_spark_df_with_null, spark=spark)


def test_process(spark, feature_pruning, features_spark_df_2):
    _, relevance_table = feature_pruning(df=features_spark_df_2, spark=spark)

    expected_relevance_table = spark.createDataFrame(
        data=[
            ("ft1", "real", 0.0, True),
            ("ft3", "real", 0.9732849094549185, False),
            ("ft2", "binary", 0.9999999999999998, False),
        ],
        schema=["feature", "type", "p_value", "relevant"],
    )

    assert_frame_equal(
        relevance_table,
        expected_relevance_table,
        check_metadata=False,
        check_types=False,
        check_column_order=False,
        check_row_order=False,
        check_nullable=False,
    )
