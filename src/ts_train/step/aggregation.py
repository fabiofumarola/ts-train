from typing import List, Tuple
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

from pydantic import BaseModel, conlist, StrictStr

from ts_train.step.core import AbstractPipelineStep
from ts_train.common.utils import *
from ts_train.common.enums import AvailableAggFunctions


class Aggregation(AbstractPipelineStep, BaseModel):
    """
    Create a new dataframe with time series. Every time series is composeb by
    aggregation of numeric and categorical values and a provided aggregation function.
    """

    numerical_col_name: conlist(StrictStr, min_length=1)  # type: ignore
    identifier_cols_name: conlist(StrictStr, min_length=1)  # type: ignore
    all_aggregation_filters: List[List[Tuple[str, list[str]]]]
    agg_funcs: conlist(AvailableAggFunctions, min_length=1)  # type: ignore

    def _preprocess(self, df: DataFrame) -> None:
        # Checks if the DataFrame is full or empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        #### CHECK IDENTIFIER COLUMNS ####

        # Convert identifier_cols_name into List[StrictStr]
        if type(self.identifier_cols_name) == str:
            self.identifier_cols_name = [self.identifier_cols_name]

        # Checks if identifier_cols_name are columns
        for identifier_col_name in self.identifier_cols_name:
            if not is_column_present(df, identifier_col_name):
                raise ValueError(f"Column {identifier_col_name} is not a column")

        #### CHECK NUMERIC COLUMNS ####

        # existency and type check on numerical columns
        for numerical_col_name in self.numerical_col_name:
            if not is_column_present(df, numerical_col_name):
                raise ValueError(f"Column {numerical_col_name} is not a column")

            if not is_column_numerical(df, numerical_col_name):
                raise ValueError(f"Column {numerical_col_name} is not a numeric column")

        #### CHECK CATEGORICAL COLUMNS ####

        # existency and type check on categorical columns
        categorical_var_names = []
        for agg_cols in self.all_aggregation_filters:
            for agg in agg_cols:
                categorical_var_names.append(agg[0])

        # existency and type check on numerical columns
        for cat_var_name in categorical_var_names:
            if not is_column_present(df, cat_var_name):
                raise ValueError(f"Column {cat_var_name} is not a column")

            if not is_column_categorical(df, cat_var_name):
                raise ValueError(f"Column {cat_var_name} is not a categorical column")

    ########  Generazione di tutte le aggregazioni
    # genero il prodotto cartesiano di tutte le possibili funzioni di aggregazioni
    # su tutte le possibili colonne numeriche
    # lista di tutti dizionari di aggregazioni possibili
    def _all_aggregation_combination(
        self, numerical_col_name: list[str], aggr_functions: list[AvailableAggFunctions]
    ) -> list[dict[str, str]]:
        all_aggregations = []
        for col in numerical_col_name:
            for func in aggr_functions:
                all_aggregations.append({col: func.value})

        return all_aggregations

    def _parse_aggregation(self, aggregation: Dict[str, str]) -> Tuple[str, str]:
        return (list(aggregation.keys())[0], list(aggregation.values())[0])

    def _rename_pivoting_cols(
        self,
        df: DataFrame,
        filter: tuple[str, list[str]],
        aggregation: dict[str, str],
    ) -> DataFrame:
        numerical_col_name, operation = self._parse_aggregation(aggregation)
        cat_var_name, cat_options = filter

        for cat_option in cat_options:
            auto_col_name = cat_option
            new_col_name = (
                f"{operation}_{numerical_col_name}_by_{cat_var_name}_({cat_option})"
            )

            df = df.withColumnRenamed(auto_col_name, new_col_name)

        return df

    def _rename_selecting_col(
        self,
        df: DataFrame,
        filters: list[tuple[str, list[str]]],
        aggregation: dict[str, str],
    ) -> DataFrame:
        numerical_col_name, operation = self._parse_aggregation(aggregation)

        auto_col_name = operation + "(" + numerical_col_name + ")"
        new_col_name = f"{operation}_of_{numerical_col_name}"

        for filter in filters:
            create_column_name, categotrical_options = filter
            if len(create_column_name) >= 0:
                new_col_name += f"_by_{create_column_name}"
            if len(categotrical_options) > 0:
                categotrical_options = [
                    str(categotrical_option)
                    for categotrical_option in categotrical_options
                ]
                new_col_name += f"_({'_'.join(categotrical_options)})"
            new_col_name += "_and"
        new_col_name = new_col_name[:-4]  # remove last "_and"

        df = df.withColumnRenamed(auto_col_name, new_col_name)

        return df

    def _pivoting(
        self,
        df: DataFrame,
        extended_id_cols_name: list[str],
        filter: tuple[str, List[str]],
        aggregation: dict[str, str],
    ) -> DataFrame:
        print("pivot")
        cat_var_name, cat_options = filter
        grouped_df = df.groupBy(*extended_id_cols_name)

        # Pre-caltulates categorical variable options
        if len(cat_options) == 0:
            cat_options = [
                str(row[0]) for row in df.select(cat_var_name).distinct().collect()
            ]
            filter = (cat_var_name, cat_options)

        # Pivotes
        pivoted_df = grouped_df.pivot(cat_var_name, cat_options)  # type: ignore
        aggregated_df = pivoted_df.agg(aggregation)

        # Renaming
        aggregated_df = self._rename_pivoting_cols(
            df=aggregated_df,
            filter=filter,
            aggregation=aggregation,
        )

        return aggregated_df

    def _selecting(
        self,
        df: DataFrame,
        extended_id_cols_name: List[str],
        filters: List[Tuple[str, List[str]]],
        aggregation: Dict[str, str],
    ) -> DataFrame:
        # Creates filter concatenation and filters the input DataFrame
        filter_conditions = []
        for single_filter in filters:
            filter_column, filter_options = single_filter
            filter_conditions.append(F.col(filter_column).isin(filter_options))

        filter_expression = reduce(lambda f1, f2: f1 & f2, filter_conditions)
        filtered_df = df.filter(filter_expression)

        # Aggregates rows using the provided aggregation function
        result_df = filtered_df.groupBy(*extended_id_cols_name).agg(aggregation)

        # Renames produced column
        result_df = self._rename_selecting_col(
            df=result_df, filters=filters, aggregation=aggregation
        )

        # Creates a DataFrame with every value equal to 0
        zero_rows_df = df.groupBy(*extended_id_cols_name).agg(
            F.lit(0).alias("zero_rows_column")
        )

        # Joins the DataFrame with values with that with 0s and drops the support col
        result_df = result_df.join(
            zero_rows_df, on=extended_id_cols_name, how="outer"
        ).drop("zero_rows_column")

        return result_df

    def _join_dataframes(self, df1: DataFrame, df2: DataFrame) -> DataFrame:
        join_columns = [col_name for col_name in df1.columns if col_name in df2.columns]

        return df1.join(df2, join_columns)

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        # all_aggregation_filters: List[List[Tuple[str, list[str]]]]
        # PYTEST:
        # possiamo calcolare a priori il numero di colonne che dovrebbe avere il df
        # risultante?

        # assert len(old_df) <= len(new_df)
        # contare il numero di nuove colonne attese e verificare che siano state
        # prodotte tutte
        # fare un caso in cui metti solo variabili nuemriche e verichi le colonne
        # generate
        all_aggregations = self._all_aggregation_combination(
            self.numerical_col_name, self.agg_funcs
        )

        extended_id_cols_name = self.identifier_cols_name + [
            "bucket"
        ]  # mettere a parametro TODO

        all_aggregated_df = []
        for aggregation in all_aggregations:
            for aggregation_filter in self.all_aggregation_filters:
                if len(aggregation_filter) > 1:
                    filtered_df = self._selecting(
                        df,
                        extended_id_cols_name,
                        aggregation_filter,
                        aggregation,
                    )
                    all_aggregated_df.append(filtered_df)
                else:
                    pivoted_df = self._pivoting(
                        df,
                        extended_id_cols_name,
                        aggregation_filter[0],
                        aggregation,
                    )
                    all_aggregated_df.append(pivoted_df)

        result_df = reduce(
            lambda df1, df2: self._join_dataframes(df1, df2),
            all_aggregated_df,
        )

        return result_df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        return result
