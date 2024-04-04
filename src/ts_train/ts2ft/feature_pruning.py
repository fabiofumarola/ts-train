from typing import *

from pydantic import BaseModel, StrictStr
from pyspark.sql import SparkSession, DataFrame
import pandas as pd
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
        target_col_name (StrictStr): name of the column used as label/target.
    """

    identifier_col_name: StrictStr
    target_col_name: StrictStr

    def _preprocess(self, df: DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Checks for problems with the input DataFrame and converts it to Pandas.

        Args:
            df (DataFrame): DataFrame with the generated features. It should be a Pandas
                DataFrame. From previoues steps you have to convert it from a Spark one
                with a toPandas method.

        Raises:
            ValueError: with message "Column {identifier_col_name} is not a column" if
                identifier_col_name is not present in the DataFrame.
            ValueError: with message "Column {target_col_name} is not a column" if
                target_col_name is not present in the DataFrame.

        Returns:
            df (pd.DataFrame): Pandas DataFrame without the identifier_col_name column
            identifier_col (pd.Series): Series representing identifier_col_name column
            target_col (pd.Series): Series representing target_col_name column
        """
        # Checks if identifier_col_name is a column
        if self.identifier_col_name not in df.columns:
            raise ValueError(f"Column {self.identifier_col_name} is not a column")

        # Checks if target_col_name is a column
        if self.target_col_name not in df.columns:
            raise ValueError(f"Column {self.target_col_name} is not a column")

        # Converts the Spark DataFrame into a Pandas DataFrame
        pandas_df = df.toPandas()

        # Checks if input DataFrame contains null values
        check_for_nans_in_columns(pandas_df)

        # Removes identifier_col_name and stores it
        identifier_col = pandas_df[self.identifier_col_name]
        pandas_df = pandas_df.drop(self.identifier_col_name, axis=1)

        # Removes target_col_nmae and stores it
        target_col = pandas_df[self.target_col_name]
        pandas_df = pandas_df.drop(self.target_col_name, axis=1)

        return pandas_df, identifier_col, target_col

    def _process(
        self, spark: SparkSession, df: pd.DataFrame, targets: pd.Series
    ) -> Tuple[pd.DataFrame, DataFrame]:
        """Prunes the original Pandas DataFrames and generates the relevance table.

        Args:
            df (pd.DataFrame): Pandas DataFrame with every features for every user.
            targets (pd.Series): Targets for every user in the same order as the Pandas
                DataFrame.

        Returns:
            pruned_df (pd.DataFrame): Pandas DataFrame containing the selected features
                which are considered relevant for the target.
            relevance_table (pd.DataFrame): A table containing p-values for each
                features of the input Pandas DataFrame telling if the feature is
                relevant for the target or not.
        """
        relevance_table_df = calculate_relevance_table(df, targets)
        relevant_features = relevance_table_df[relevance_table_df.relevant].feature

        pruned_df = df.loc[:, relevant_features]
        relevance_table_spark_df = spark.createDataFrame(relevance_table_df)

        return pruned_df, relevance_table_spark_df

    def _postprocess(
        self,
        spark: SparkSession,
        df: pd.DataFrame,
        identifier_col: pd.Series,
        target_col: pd.Series,
    ) -> DataFrame:
        """Re-attaches the identifier and label columns to the resulting Pandas
        DataFrame.

        Args:
            df (pd.DataFrame): Pandas DataFrame with selected features.
            identifier_col (pd.Series): Column with identification for the users.

        Returns:
            DataFrame: Spark DataFrame with selected features and attached
                identification for every user.
        """
        df.insert(0, self.target_col_name, target_col)
        df.insert(0, self.identifier_col_name, identifier_col)

        spark_df = spark.createDataFrame(df)

        return spark_df

    def __call__(
        self, df: DataFrame, spark: SparkSession
    ) -> Tuple[DataFrame, DataFrame]:
        """Executes the _preprocess, _process and _postprocess methods.

        Args:
            df (DataFrame): DataFrame with a row for every user and a column for every
                feature. It should be a Pandas DataFrame, so from the previous step you
                have to convert the Spark DataFrame to a Pandas one using the toPandas
                method.

        Returns:
            pruned_df (DataFrame): DataFrame containing the selected features
                which are considered relevant for the target.
            relevance_table (DataFrame): A table containing p-values for each features
                of the input DataFrame telling if the feature is relevant for the target
                or not.
        """
        pandas_df, identifier_col, target_col = self._preprocess(df)
        pruned_df, relevance_table = self._process(spark, pandas_df, target_col)
        spark_pruned_df = self._postprocess(
            spark, pruned_df, identifier_col, target_col
        )

        return spark_pruned_df, relevance_table
