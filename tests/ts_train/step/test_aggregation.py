import pytest
from pyspark_assert import assert_frame_equal  # type: ignore

from ts_train.step.aggregation import Aggregation  # type: ignore


# TESTING INIT METHOD


def test_init_emptiness(sample_dataframe_empty, standard_aggregation):
    """Tests if _preprocess method raises ValueError("Empty DataFrame") in case you
    provide an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Empty DataFrame"):
        standard_aggregation._preprocess(sample_dataframe_empty)


# TESTING _PREPROCESS METHOD


def test_preprocess_wrong_numerical_column_name(sample_dataframe_01):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    numerical_col_name = ["IMPORTO", "wrong_name"]
    agg_step = Aggregation(
        numerical_col_name=numerical_col_name,
        identifier_cols_name=["ID_BIC_CLIENTE"],
        all_aggregation_filters=[],
        agg_funcs=["sum"],
    )

    with pytest.raises(
        ValueError,
        match="Column wrong_name is not a column",
    ):
        agg_step._preprocess(sample_dataframe_01)


def test_preprocess_not_numerical_column_name(sample_dataframe_01):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    numerical_col_name = ["IMPORTO", "CA_CATEGORY_LIV0"]
    agg_step = Aggregation(
        numerical_col_name=numerical_col_name,
        identifier_cols_name=["ID_BIC_CLIENTE"],
        all_aggregation_filters=[],
        agg_funcs=["sum"],
    )

    with pytest.raises(
        ValueError,
        match="Column CA_CATEGORY_LIV0 is not a numeric column",
    ):
        agg_step._preprocess(sample_dataframe_01)


def test_preprocess_wrong_categorical_column_name(sample_dataframe_01):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    filter = [[("CA_CATEGORY_LIV0", ["shopping"]), ("wrong_col_name", [])]]
    agg_step = Aggregation(
        numerical_col_name=["IMPORTO"],
        identifier_cols_name=["ID_BIC_CLIENTE"],
        all_aggregation_filters=filter,
        agg_funcs=["sum"],
    )

    with pytest.raises(
        ValueError,
        match="Column wrong_col_name is not a column",
    ):
        agg_step._preprocess(sample_dataframe_01)


def test_preprocess_not_categorical_column_name(sample_dataframe_01):
    """Tests if __init__ method raises ValidationError (made by Pydantic) if
    time_bucket_size not of the right type (int) or of value <= 0.
    """
    filter = [[("IMPORTO", []), ("CA_CATEGORY_LIV0", ["shopping"])]]
    agg_step = Aggregation(
        numerical_col_name=["IMPORTO"],
        identifier_cols_name=["ID_BIC_CLIENTE"],
        all_aggregation_filters=filter,
        agg_funcs=["sum"],
    )

    with pytest.raises(
        ValueError,
        match="Column IMPORTO is not a categorical column",
    ):
        agg_step._preprocess(sample_dataframe_01)


# TESTING _PROCESS METHOD


@pytest.mark.parametrize(
    "filter, expected_df",
    [
        (("CA_CATEGORY_LIV0", []), "result_aggregation_pivot_string_cat_no_options"),
        (
            ("CA_CATEGORY_LIV0", ["salute"]),
            "result_aggregation_pivot_string_cat_one_option",
        ),
        (
            ("CA_CATEGORY_LIV0", ["salute", "trasporti"]),
            "result_aggregation_pivot_string_cat_two_options",
        ),
        (
            ("IS_CARTA", []),
            "result_aggregation_pivot_bool_cat_no_options",
        ),
    ],
)
def test_pivoting(
    sample_dataframe_01_bucketed,
    standard_aggregation: Aggregation,
    filter,
    expected_df,
    request,
):
    """Tests if _pivoting method works as expeced. Testing aggregation of variable
    IMPORTO using sum operation grouping using ID_BIC_CLIENTE and bucket columns.
    As filters we are testing different filters with related different expected dfs."""
    result_df = standard_aggregation._pivoting(
        df=sample_dataframe_01_bucketed,
        extended_id_cols_name=["ID_BIC_CLIENTE", "bucket"],
        filter=filter,
        aggregation={"IMPORTO": "sum"},
    )

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
            [("CA_CATEGORY_LIV0", ["salute", "trasporti"]), ("IS_CARTA", [True])],
            "result_aggregation_select_salute_trasporti_is_carta",
        ),
        (
            [("CA_CATEGORY_LIV0", ["salute"]), ("IS_CARTA", [True])],
            "result_aggregation_select_salute_is_carta",
        ),
        (
            [("CA_CATEGORY_LIV0", ["salute", "trasporti"])],
            "result_aggregation_select_salute_trasporti",
        ),
    ],
)
def test_selecting(
    sample_dataframe_01_bucketed,
    standard_aggregation: Aggregation,
    filters,
    expected_df,
    request,
):
    """Tests if _selecting method works as expeced. Testing aggregation of variable
    IMPORTO using sum operation grouping using ID_BIC_CLIENTE and bucket columns.
    As filters we are testing different filters with related different expected dfs."""
    result_df = standard_aggregation._selecting(
        df=sample_dataframe_01_bucketed,
        extended_id_cols_name=["ID_BIC_CLIENTE", "bucket"],
        filters=filters,
        aggregation={"IMPORTO": "sum"},
    )

    expected_df = request.getfixturevalue(expected_df)

    sample_dataframe_01_bucketed.show()
    result_df.show()
    expected_df.show()

    assert_frame_equal(
        result_df,
        expected_df,
        check_column_order=False,
        check_row_order=False,
        check_nullable=False,
    )


# @pytest.mark.parametrize(
#     "all_aggregation_filters, expected_df",
#     [
#         (
#             [
#                 [("CA_CATEGORY_LIV0", ["salute", "trasporti"])],
#                 [("CA_CATEGORY_LIV0", ["salute"])],
#             ],
#             "result_aggregation_process_01",
#         ),
#         # (
#         #     [
#         #         [("CA_CATEGORY_LIV0", ["salute"])],
#         #         [("CA_CATEGORY_LIV0", [])],
#         #     ],
#         #     "",
#         # ),
#         # (
#         #     [
#         #         [("CA_CATEGORY_LIV0", ["salute", "trasporti"]), ("IS_CARTA", [True])
# ],
#         #         [("CA_CATEGORY_LIV0", ["salute"])],
#         #     ],
#         #     "",
#         # ),
#     ],
# )
# def test_process(
#     spark,
#     sample_dataframe_01_bucketed,
#     standard_aggregation: Aggregation,
#     all_aggregation_filters,
#     expected_df,
#     request,
# ):
#     standard_aggregation.all_aggregation_filters = all_aggregation_filters

#     result_df = standard_aggregation._process(
#         df=sample_dataframe_01_bucketed, spark=spark
#     )

#     expected_df = request.getfixturevalue(expected_df)

#     sample_dataframe_01_bucketed.show()
#     result_df.show()
#     expected_df.show()

#     assert_frame_equal(
#         result_df,
#         expected_df,
#         check_column_order=False,
#         check_row_order=False,
#         check_nullable=False,
#     )
