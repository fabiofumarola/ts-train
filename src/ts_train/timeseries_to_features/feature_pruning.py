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
    """FeaturePruning step, being part of the second block called ts2ft, is used to
    select features which are relevant for downstream tasks to predict or model a target
    variable. For this reason in this step we need the DataFrame with every feature
    generated with the FeatureGenerating step for each user and a target for every user.

    Attributes:
        identifier_col_name (StrictStr): name of the column used to identify users.
    """

    identifier_col_name: StrictStr

    def _preprocess(self, df: DataFrame, targets: Series) -> Tuple[DataFrame, Series]:
        """Checks for problems with the input DataFrame and the Series.

        Args:
            df (DataFrame): DataFrame with the generated features. It should be a Pandas
                DataFrame. From previoues steps you have to convert it from a Spark one
                with a toPandas method.
            targets (Series): list of targets to be used to understand if the features
                are relevent for the task you want to perform after this block of steps.

        Raises:
            ValueError: with message "Column {identifier_col_name} is not a column" if
                identifier_col_name is not present in the DataFrame.
            ValueError: with message "df should be a pandas.DataFrame" if the provided
                DataFrame is not a Pandas DataFrame.
            ValueError: with message "targets should be a pandas.Series" if the provided
                targets is not a Pandas Series.
            ValueError: with message "df and targets have different length" if the
                DataFrame and the Series have different lengths. Remember that you have
                to have a target for every user, so lengths should be the same.

        Returns:
            df (DataFrame): DataFrame without the identifier_col_name column
            identifier_col (Series): Series representing identifier_col_name column
        """
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
        """Prunes the original DataFrames and generates the relevance table.

        Args:
            df (DataFrame): DataFrame with every features for every user.
            targets (Series): Targets for every user in the same order as the DataFrame.

        Returns:
            pruned_df (DataFrame): Pandas DataFrame containing the selected features
                which are considered relevant for the target.
            relevance_table (DataFrame): A table containing p-values for each features
                of the input DataFrame telling if the feature is relevant for the target
                or not.
        """
        relevance_table = calculate_relevance_table(df, targets)
        relevant_features = relevance_table[relevance_table.relevant].feature

        pruned_df = df.loc[:, relevant_features]

        return pruned_df, relevance_table

    def _postprocess(self, df: DataFrame, identifier_col: Series) -> DataFrame:
        """Re-attaches the identifier column to the resulting DataFrame.

        Args:
            df (DataFrame): DataFrame with selected features.
            identifier_col (Series): Column with identification for the users.

        Returns:
            DataFrame: DataFrame with selected features and attached identification for
                every user.
        """
        df.insert(0, self.identifier_col_name, identifier_col)

        return df

    def __call__(self, df: DataFrame, targets: Series) -> Tuple[DataFrame, DataFrame]:
        """Executes the _preprocess, _process and _postprocess methods.

        Args:
            df (DataFrame): DataFrame with a row for every user and a column for every
                feature. It should be a Pandas DataFrame, so from the previous step you
                have to convert the Spark DataFrame to a Pandas one using the toPandas
                method.
            targets (Series): This is the list of the targets for every user. It should
                be a Pandas Series.

        Returns:
            pruned_df (DataFrame): Pandas DataFrame containing the selected features
                which are considered relevant for the target.
            relevance_table (DataFrame): A table containing p-values for each features
                of the input DataFrame telling if the feature is relevant for the target
                or not.
        """
        df, identifier_col = self._preprocess(df, targets)
        pruned_df, relevance_table = self._process(df, targets)
        pruned_df = self._postprocess(pruned_df, identifier_col)

        return pruned_df, relevance_table
