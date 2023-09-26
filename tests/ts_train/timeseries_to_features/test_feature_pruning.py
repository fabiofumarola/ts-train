import pytest
import pandas as pd

from ts_train.timeseries_to_features.feature_pruning import FeaturePruning  # type: ignore


@pytest.fixture
def feature_pruning():
    return FeaturePruning(identifier_col_name="ID_CLIENTE_BIC")


@pytest.fixture
def features_spark_df(spark):
    df = spark.createDataFrame(
        data=[
            (1, 1, 33),
            (2, 10, 56),
            (3, 100, 12),
        ],
        schema=[
            "ID_CLIENTE_BIC",
            "ft1",
            "ft2",
        ],
    )

    return df


@pytest.fixture
def features_pandas_df():
    return pd.DataFrame(
        {
            "ID_CLIENTE_BIC": [1, 2, 3],
            "ft1": [1, 10, 100],
            "ft2": [33, 56, 12],
        }
    )


@pytest.fixture
def features_pandas_df_with_null():
    return pd.DataFrame(
        {
            "ID_CLIENTE_BIC": [1, 2, 3],
            "ft1": [1, None, 100],
            "ft2": [33, 56, 12],
        }
    )


@pytest.fixture
def features_pandas_df_2(targets_2):
    return pd.DataFrame(
        {
            "ID_CLIENTE_BIC": range(0, 1000),
            "ft1": [num * 2 for num in targets_2],
            "ft2": [num % 2 for num in range(0, 1000)],
            "ft3": [num % 3 for num in range(0, 1000)],
        }
    )


# @pytest.fixture
# def expected_relevance_table():
#     return pd.DataFrame(
#         {
#             "ID_CLIENTE_BIC": [1, 2, 3],
#             "ft1": [1, 10, 100],
#         }
#     )


@pytest.fixture
def targets():
    return pd.Series([1, 10, 100])


@pytest.fixture
def targets_shorter():
    return pd.Series([2, 20])


@pytest.fixture
def targets_2():
    return pd.Series([num + 3.2 for num in range(0, 1000)])


def test_preprocess_check_pandas_pd(feature_pruning, features_spark_df, targets):
    with pytest.raises(ValueError, match="df should be a pandas.DataFrame"):
        feature_pruning(df=features_spark_df, targets=targets)


def test_preprocess_check_targets(feature_pruning, features_pandas_df):
    with pytest.raises(ValueError, match="targets should be a pandas.Series"):
        feature_pruning(df=features_pandas_df, targets=[2, 20, 200])


def test_preprocess_check_null_in_columns(
    feature_pruning, features_pandas_df_with_null, targets
):
    with pytest.raises(
        ValueError, match="Columns \['ft1'\] of DataFrame must not contain NaN values"
    ):
        feature_pruning(df=features_pandas_df_with_null, targets=targets)


def test_preprocess_check_len_of_df_and_targets(
    feature_pruning, features_pandas_df, targets_shorter
):
    with pytest.raises(ValueError, match="df and targets have different length"):
        feature_pruning(df=features_pandas_df, targets=targets_shorter)


def test_process(feature_pruning, features_pandas_df_2, targets_2):
    _, relevance_table = feature_pruning(df=features_pandas_df_2, targets=targets_2)

    pd.testing.assert_series_equal(
        pd.Series(list(relevance_table["p_value"])),
        pd.Series([0.000000, 0.973285, 1.000000]),
    )

    pd.testing.assert_series_equal(
        pd.Series(list(relevance_table["relevant"])),
        pd.Series([True, False, False]),
    )
