import pytest
from pyspark_assert import assert_frame_equal  # type: ignore
from pyspark.sql.types import StructType, StructField, LongType, DoubleType

from ts_train.timeseries_to_features.feature_generating import FeatureGenerating  # type: ignore
from ts_train.common.utils import (  # type: ignore
    cast_column_to_timestamp,
)


@pytest.fixture
def filled_dataframe(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, "2023-01-01", 5.50, 1),
            (348272371, "2023-01-02", 0.0, 3),
            (348272371, "2023-01-03", 8.20, 0),
            (348272371, "2023-01-04", 0.0, 0),
            (348272371, "2023-01-05", 0.0, 0),
            (348272371, "2023-01-06", 0.0, 0),
            (348272371, "2023-01-03", 72.20, 4),
            (234984832, "2023-01-04", 15.34, 5),
            (234984832, "2023-01-05", 0.0, 6),
            (234984832, "2023-01-06", 78.35, 9),
            (234984832, "2023-01-07", 2.20, 11),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "agg1",
            "agg2",
        ],
    )

    return cast_column_to_timestamp(df=df, col_name="bucket_start")


@pytest.fixture
def filled_dataframe_with_a_not_numerical_feature(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, "2023-01-01", "a", 5.50, 1),
            (348272371, "2023-01-02", "a", 0.0, 3),
            (348272371, "2023-01-03", "b", 8.20, 0),
            (348272371, "2023-01-04", "a", 0.0, 0),
            (348272371, "2023-01-05", "a", 0.0, 0),
            (348272371, "2023-01-06", "b", 0.0, 0),
            (348272371, "2023-01-03", "b", 72.20, 4),
            (234984832, "2023-01-04", "b", 15.34, 5),
            (234984832, "2023-01-05", "b", 0.0, 6),
            (234984832, "2023-01-06", "a", 78.35, 9),
            (234984832, "2023-01-07", "a", 2.20, 11),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "agg1",
            "agg2",
            "agg3",
        ],
    )

    return cast_column_to_timestamp(df=df, col_name="bucket_start")


@pytest.fixture
def expected_stacked_df(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, "2023-01-01", "agg1", 5.50),
            (348272371, "2023-01-01", "agg2", 1.0),
            (348272371, "2023-01-02", "agg1", 0.0),
            (348272371, "2023-01-02", "agg2", 3.0),
            (348272371, "2023-01-03", "agg1", 8.20),
            (348272371, "2023-01-03", "agg2", 0.0),
            (348272371, "2023-01-04", "agg1", 0.0),
            (348272371, "2023-01-04", "agg2", 0.0),
            (348272371, "2023-01-05", "agg1", 0.0),
            (348272371, "2023-01-05", "agg2", 0.0),
            (348272371, "2023-01-06", "agg1", 0.0),
            (348272371, "2023-01-06", "agg2", 0.0),
            (348272371, "2023-01-03", "agg1", 72.20),
            (348272371, "2023-01-03", "agg2", 4.0),
            (234984832, "2023-01-04", "agg1", 15.34),
            (234984832, "2023-01-04", "agg2", 5.0),
            (234984832, "2023-01-05", "agg1", 0.0),
            (234984832, "2023-01-05", "agg2", 6.0),
            (234984832, "2023-01-06", "agg1", 78.35),
            (234984832, "2023-01-06", "agg2", 9.0),
            (234984832, "2023-01-07", "agg1", 2.20),
            (234984832, "2023-01-07", "agg2", 11.0),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "bucket_start",
            "kind",
            "value",
        ],
    )

    return cast_column_to_timestamp(df=df, col_name="bucket_start")


@pytest.fixture
def expected_features_df(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, 72.20, 4.0),
            (234984832, 78.35, 11.0),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "agg1__absolute_maximum",
            "agg2__absolute_maximum",
        ],
    )

    return df


@pytest.fixture
def features_df(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, 72.20, None),
            (234984832, 78.35, None),
        ],
        schema=StructType(
            [
                StructField("ID_BIC_CLIENTE", LongType(), True),
                StructField("agg1__absolute_maximum_param_0.0", DoubleType(), True),
                StructField("agg2__absolute_maximum", DoubleType(), True),
            ]
        ),
    )

    return df


@pytest.fixture
def expected_cleaned_features_df(spark):
    df = spark.createDataFrame(
        data=[
            (348272371, 72.20),
            (234984832, 78.35),
        ],
        schema=[
            "ID_BIC_CLIENTE",
            "agg1__absolute_maximum_param_0dot0",
        ],
    )

    return df


@pytest.fixture
def standard_feature_generating():
    return FeatureGenerating(
        identifier_col_name="ID_BIC_CLIENTE",
        time_col_name="bucket_start",
        feature_calculators=["mean_n_absolute_max", "benford_correlation"],
    )


# Testing _preprocess


def test_preprocess_check_feature_calculators(
    standard_feature_generating, filled_dataframe
):
    standard_feature_generating.feature_calculators = ["benford_correlation_typo"]

    with pytest.raises(
        ValueError,
        match="Feature calculator benford_correlation_typo is not supported",
    ):
        standard_feature_generating._preprocess(filled_dataframe)


def test_preprocess_get_default_fc_parameters(
    standard_feature_generating, filled_dataframe
):
    standard_feature_generating._preprocess(filled_dataframe)

    default_fc_parameters = standard_feature_generating._default_fc_parameters

    assert (
        len(default_fc_parameters)
        == len(standard_feature_generating.feature_calculators)
        and default_fc_parameters["mean_n_absolute_max"] == [{"number_of_maxima": 7}]
        and default_fc_parameters["benford_correlation"] is None
    )


def test_preprocess_check_feature_columns(
    standard_feature_generating, filled_dataframe_with_a_not_numerical_feature
):
    with pytest.raises(ValueError, match="Column agg1 is not a numerical column"):
        standard_feature_generating._preprocess(
            filled_dataframe_with_a_not_numerical_feature
        )


def test_stack_df(standard_feature_generating, filled_dataframe, expected_stacked_df):
    stacked_df = standard_feature_generating._stack_df(filled_dataframe)

    assert_frame_equal(
        stacked_df, expected_stacked_df, check_nullable=False, check_row_order=False
    )


def test_generate_features(
    standard_feature_generating: FeatureGenerating,
    expected_stacked_df,
    expected_features_df,
):
    standard_feature_generating.feature_calculators = ["absolute_maximum"]

    standard_feature_generating._default_fc_parameters = (
        standard_feature_generating._get_default_fc_parameters()
    )
    features_df = standard_feature_generating._generate_features(expected_stacked_df)

    features_df.show()

    assert_frame_equal(features_df, expected_features_df)


def test_clean_features_df(
    standard_feature_generating: FeatureGenerating,
    features_df,
    expected_cleaned_features_df,
):
    standard_feature_generating.feature_calculators = ["absolute_maximum"]

    cleaned_features_df = standard_feature_generating._clean_features_df(features_df)

    assert_frame_equal(cleaned_features_df, expected_cleaned_features_df)


def test_process(
    standard_feature_generating: FeatureGenerating,
    filled_dataframe,
    expected_features_df,
):
    standard_feature_generating.feature_calculators = ["absolute_maximum"]

    features_df = standard_feature_generating(filled_dataframe)

    features_df.show()

    assert_frame_equal(features_df, expected_features_df)
