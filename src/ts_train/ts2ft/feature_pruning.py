from typing import *

from pydantic import BaseModel, StrictStr
from pandas import DataFrame, Series
from tsfresh.feature_selection.relevance import (  # type: ignore
    calculate_relevance_table,
)
from tsfresh.utilities.dataframe_functions import (  # type: ignore
    check_for_nans_in_columns,
)


class FeaturePruning(BaseModel):
    identifier_col_name: StrictStr

    def _preprocess(self, df: DataFrame, targets: Series) -> Tuple[DataFrame, Series]:
        # Checks if identifier_col_name is a column
        if self.identifier_col_name not in df.columns:
            raise ValueError(f"Column {self.identifier_col_name} is not a column")

        # Checks if input DataFrame is a Pandas DataFrame
        if not isinstance(df, DataFrame):
            raise ValueError("df should be a pandas.DataFrame")

        # Checks if input DataFrame contains null values
        check_for_nans_in_columns(df)

        # Checks if input targets is a Pandas Series
        if not isinstance(targets, Series):
            raise ValueError("targets should be a pandas.Series")

        # Checks if length of df and targets are the same
        if len(df) != len(targets):
            raise ValueError("df and targets have different length")

        # Removes identifier_col_name and stores it
        identifier_col = df[self.identifier_col_name]
        df = df.drop(self.identifier_col_name, axis=1)

        return df, identifier_col

    def _process(self, df: DataFrame, targets: Series) -> Tuple[DataFrame, DataFrame]:
        relevance_table = calculate_relevance_table(df, targets)
        relevant_features = relevance_table[relevance_table.relevant].feature

        pruned_df = df.loc[:, relevant_features]

        return pruned_df, relevance_table

    def _postprocess(self, df: DataFrame, identifier_col: Series) -> DataFrame:
        df.insert(0, self.identifier_col_name, identifier_col)

        return df

    def __call__(self, df: DataFrame, targets: Series) -> Tuple[DataFrame, DataFrame]:
        df, identifier_col = self._preprocess(df, targets)
        pruned_df, relevance_table = self._process(df, targets)
        pruned_df = self._postprocess(pruned_df, identifier_col)

        return pruned_df, relevance_table
