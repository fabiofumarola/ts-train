import pytest
from pyspark_assert import assert_frame_equal  # type: ignore

from ts_train.tr2ts.aggregating import (  # type: ignore
    Aggregating,  # type: ignore
    Aggregation,  # type: ignore
    Filter,  # type: ignore
    Pivot,  # type: ignore
)  # type: ignore

# TESTING INIT METHOD


@pytest.mark.slow
def test_init_emptiness(sample_dataframe_empty, standard_aggregator):
    """Tests if _preprocess method raises ValueError("Empty DataFrame") in case you
    provide an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_aggregator._preprocess(sample_dataframe_empty)


# TESTING _PREPROCESS METHOD


def test_preprocess_wrong_numerical_column_name(sample_dataframe_01_bucketed):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    agg_step = Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(
                numerical_col_name="wrong_name",
                agg_function="sum",
                filters=[Filter(col_name="IS_CARTA", operator="==", value=True)],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Column wrong_name is not a column",
    ):
        agg_step._preprocess(sample_dataframe_01_bucketed)


def test_preprocess_not_numerical_column_name(sample_dataframe_01_bucketed):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    agg_step = Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(
                numerical_col_name="CA_CATEGORY_LIV0",
                agg_function="sum",
                filters=[Filter(col_name="IS_CARTA", operator="==", value=True)],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Column CA_CATEGORY_LIV0 is not a numeric column",
    ):
        agg_step._preprocess(sample_dataframe_01_bucketed)


def test_preprocess_wrong_categorical_column_name(sample_dataframe_01_bucketed):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    agg_step = Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                filters=[Filter(col_name="wrong_col_name", operator="==", value=True)],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Column wrong_col_name is not a column",
    ):
        agg_step._preprocess(sample_dataframe_01_bucketed)


def test_preprocess_not_bool_value_for_bool_col(
    sample_dataframe_01_bucketed,
):
    agg_step = Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(
                numerical_col_name="IMPORTO",
                agg_function="sum",
                filters=[Filter(col_name="IS_CARTA", operator="==", value=4)],
            )
        ],
    )

    sample_dataframe_01_bucketed.printSchema()

    with pytest.raises(
        ValueError,
        match="Mismatch types between IS_CARTA column *",
    ):
        agg_step._preprocess(sample_dataframe_01_bucketed)


# TESTING _PROCESS METHOD


def test_process_no_pivot_no_filter(
    spark,
    sample_dataframe_01_bucketed,
    no_pivot_no_filter_expected_df,
):
    aggregating = Aggregating(
        identifier_cols_name=["ID_BIC_CLIENTE"],
        time_bucket_cols_name=["bucket_start", "bucket_end"],
        aggregations=[
            Aggregation(numerical_col_name="IMPORTO", agg_function="sum", filters=[])
        ],
    )

    result_df = aggregating(df=sample_dataframe_01_bucketed, spark=spark)

    assert_frame_equal(
        result_df,
        no_pivot_no_filter_expected_df,
        check_column_order=False,
        check_row_order=False,
        check_nullable=False,
    )


@pytest.mark.parametrize(
    "pivot, expected_df",
    [
        (
            Pivot("CA_CATEGORY_LIV0", "in", None),
            "result_aggregation_pivot_string_cat_no_options",
        ),
        (
            Pivot("CA_CATEGORY_LIV0", "in", ["salute"]),
            "result_aggregation_pivot_string_cat_one_option",
        ),
        (
            Pivot("CA_CATEGORY_LIV0", "in", ["salute", "trasporti"]),
            "result_aggregation_pivot_string_cat_two_options",
        ),
        (
            Pivot("IS_CARTA", "in", None),
            "result_aggregation_pivot_bool_cat_no_options",
        ),
        (
            Pivot("IS_CARTA", "in", [True]),
            "result_aggregation_pivot_bool_cat_true",
        ),
    ],
)
def test_process_only_pivot(
    spark,
    sample_dataframe_01_bucketed,
    standard_aggregator: Aggregating,
    pivot,
    expected_df,
    request,
):
    """Tests if _pivoting method works as expeced. Testing aggregator of variable
    IMPORTO using sum operation grouping using ID_BIC_CLIENTE and bucket columns.
    As filters we are testing different filters with related different expected dfs."""
    standard_aggregator.aggregations = [
        Aggregation(
            numerical_col_name="IMPORTO",
            agg_function="sum",
            pivot=pivot,
        )
    ]

    result_df = standard_aggregator(df=sample_dataframe_01_bucketed, spark=spark)
    expected_df = request.getfixturevalue(expected_df)

    assert_frame_equal(
        result_df,
        expected_df,
        check_column_order=False,
        check_row_order=False,
        check_nullable=False,
    )


@pytest.mark.parametrize(
    "filters, expected_df",
    [
        (
            [
                Filter("CA_CATEGORY_LIV0", "in", ["salute", "trasporti"]),
                Filter("IS_CARTA", "=", True),
            ],
            "result_aggregator_filters_salute_trasporti_is_carta",
        ),
        (
            [
                Filter("CA_CATEGORY_LIV0", "=", "salute"),
                Filter("IS_CARTA", "=", True),
            ],
            "result_aggregator_filters_salute_is_carta",
        ),
        (
            [Filter("CA_CATEGORY_LIV0", "in", ["salute", "trasporti"])],
            "result_aggregator_filters_salute_trasporti",
        ),
    ],
)
def test_filter(
    spark,
    sample_dataframe_01_bucketed,
    standard_aggregator: Aggregating,
    filters,
    expected_df,
    request,
):
    """Tests if _selecting method works as expeced. Testing aggregator of variable
    IMPORTO using sum operation grouping using ID_BIC_CLIENTE and bucket columns.
    As filters we are testing different filters with related different expected dfs."""
    standard_aggregator.aggregations = [
        Aggregation(numerical_col_name="IMPORTO", agg_function="sum", filters=filters)
    ]

    result_df = standard_aggregator(df=sample_dataframe_01_bucketed, spark=spark)
    expected_df = request.getfixturevalue(expected_df)

    assert_frame_equal(
        result_df,
        expected_df,
        check_column_order=False,
        check_row_order=False,
        check_nullable=False,
    )


def test_process(
    spark,
    sample_dataframe_for_aggregation_process_bucketed,
    expected_aggregated_df_01,
    complex_aggregator: Aggregating,
):
    sample_dataframe_for_aggregation_process_bucketed.show()

    result_df = complex_aggregator(
        df=sample_dataframe_for_aggregation_process_bucketed, spark=spark
    ).fillna(0)

    expected_aggregated_df_01.show()
    result_df.show()

    assert_frame_equal(
        result_df,
        expected_aggregated_df_01,
        check_metadata=False,
        check_types=False,
        check_column_order=False,
        check_row_order=False,
        check_nullable=False,
    )
