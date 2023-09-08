from typing import *

from pydantic import BaseModel, StrictStr
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from tsfresh.convenience.bindings import (  # type: ignore
    spark_feature_extraction_on_chunk,
)
from tsfresh.feature_extraction import ComprehensiveFCParameters  # type: ignore

from ts_train.common.utils import (
    is_column_present,
    is_column_timestamp,
    is_column_numerical,
)


class FeatureGenerating(BaseModel):
    """FeatureGenerating step, being part of the second block called ts2ft, is used to
    generate features from time series. You have to provide equidistant time series in
    time and you get a DataFrame with a row for every user and a column for every
    requested feature.

    Remembder to drop any column which is not a aggregation column or
    identifier_col_name or time_col_name column.

    Attributes:
        identifier_col_name (StrictStr): name of the column used to identify users.
        time_col_name (StrictStr): name of the column used to store data. You only need
            the bucket_start or the bucket_end. Drop one of the two and provide the
            other here.
        feature_calculators: list of names of feature calculators. Feature calculators
            are method to calculate features. Each feature calculator can have some
            parameters. Each feature calculator has one or more value for each
            parameter. This translates to a number of combinations generating different
            features. The list of every feature available is here:
            https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
    """

    identifier_col_name: StrictStr
    time_col_name: StrictStr
    feature_calculators: Optional[list[StrictStr]] = None

    def _get_default_fc_parameters(self) -> dict[str, Any]:
        """Retrives the filered dictionary of params for each feature calculator.

        Returns:
            dict[str, Any]: Dict with key the name of the feature calculator and with
                value the dict with the params.
        """
        default_fc_settings = dict(ComprehensiveFCParameters())
        if self.feature_calculators is None:
            return default_fc_settings

        filtered_default_fc_parameters = {}
        for fc_name, fc_parameters in default_fc_settings.items():
            if fc_name in self.feature_calculators:
                filtered_default_fc_parameters[fc_name] = fc_parameters

        return filtered_default_fc_parameters

    def _check_feature_calculators(self) -> None:
        """Checks if a feature caulculator requested is available.

        Raises:
            ValueError: with message "Feature calculator {fc_name} is not supported" if
                the feature calculator is not present in the the full list.
        """
        if self.feature_calculators is not None:
            default_fc_settings = dict(ComprehensiveFCParameters())
            for fc_name in self.feature_calculators:
                if fc_name not in default_fc_settings.keys():
                    raise ValueError(f"Feature calculator {fc_name} is not supported")

    def _preprocess(self, df: DataFrame) -> None:
        """Checks for problems with the DataFrame and the parameters provided.

        Args:
            df (DataFrame): DataFrame on which to perform the tests.

        Raises:
            ValueError: with message "Column {identifier_col_name} is not a column" when
                identifier_col_name is not present in the DataFrame.
            ValueError: with message "Column {time_col_name} is not a column" when
                time_col_name is not present in the DataFrame.
            ValueError: with message "Column {time_col_name} is not a timestamp" when
                time_col_name is not a timestamp column containing time data.
            ValueError: with message "Column {col_name} is not a numerical column" when
                other columns a part from identifier_col_name and time_col_name are not
                numerical columns.
        """
        # Checks if identifier_col_name is a column
        if not is_column_present(df, self.identifier_col_name):
            raise ValueError(f"Column {self.identifier_col_name} is not a column")

        # Checks if time_col_name is a column and is a timestamp
        if not is_column_present(df, self.time_col_name):
            raise ValueError(f"Column {self.time_col_name} is not a column")

        if not is_column_timestamp(df, self.time_col_name):
            raise ValueError(f"Column {self.time_col_name} is not a timestamp")

        # Checks if feature_calculators provided are valid feature calculator of tsfresh
        self._check_feature_calculators()

        # Retrieves default parameters for feature calculators
        self._default_fc_parameters = self._get_default_fc_parameters()

        # Checks if feature columns are numerical
        for col_name in df.columns:
            if col_name not in [self.identifier_col_name, self.time_col_name]:
                if not is_column_numerical(df, col_name):
                    raise ValueError(f"Column {col_name} is not a numerical column")

    def _stack_df(self, df: DataFrame) -> DataFrame:
        """Stacks numerical variables into a stacked format from a wide format. This is
        usefull beacuse TsFresh library needs it in this form. The resulting DataFrame
        is made of two columns:
            - kind: the name of the numerical variable
            - value: the value of that numerical variable

        Args:
            df (DataFrame): DataFram on which to perform the transformation.

        Returns:
            DataFrame: DataFrame in stacked form.
        """
        # Column names used for identification for grouping or pivoting operation
        extended_identifier_cols_name = [self.identifier_col_name] + [
            self.time_col_name
        ]

        # Column names not used for identification for grouping or pivoting operation
        not_extended_identifier_cols_name = list(
            set(df.columns) - set(extended_identifier_cols_name)
        )

        # Unpivots the DataFrame adding king and value columns
        stacked_df = df.unpivot(
            extended_identifier_cols_name,  # type: ignore
            not_extended_identifier_cols_name,  # type: ignore
            variableColumnName="kind",
            valueColumnName="value",
        )

        # Casts feature values to Double to avoid exceptions
        stacked_df = stacked_df.withColumn("value", stacked_df.value.cast(DoubleType()))

        return stacked_df

    def _generate_features(self, stacked_df: DataFrame) -> DataFrame:
        """Generates features from the stacked DataFrame. Internally we convert the
        output DataFrame with generated features from a stacked form to a wide form.

        Args:
            stacked_df (DataFrame): Stacked DataFrame on which to generate features.

        Returns:
            DataFrame: DataFrame in wide form with generated features.
        """
        grouped_stacked_df = stacked_df.groupby(self.identifier_col_name, "kind")

        # Calculates features on spark on chunks in a stacked format
        stacked_features_df = spark_feature_extraction_on_chunk(
            grouped_stacked_df,
            column_id=self.identifier_col_name,
            column_kind="kind",
            column_value="value",
            column_sort=self.time_col_name,
            default_fc_parameters=self._default_fc_parameters,
        )

        # Unstackes generated features
        unstacked_features_df = stacked_features_df.groupby(
            self.identifier_col_name
        ).pivot("variable")
        unstacked_features_df = unstacked_features_df.agg(F.first("value"))

        return unstacked_features_df

    def _drop_null_columns(self, df: DataFrame) -> DataFrame:
        """Drops columns with every value null. This features are not useful.

        Args:
            df (DataFrame): DataFrame on which to perform the drop operation.

        Returns:
            DataFrame: DataFrame without
        """
        null_counts = (
            df.select(
                [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
            )
            .collect()[0]
            .asDict()
        )
        to_drop = [k for k, v in null_counts.items() if v > 0]
        df = df.drop(*to_drop)

        return df

    def _clean_features_df(self, features_df: DataFrame) -> DataFrame:
        """Cleans the DataFrame. It renames columns replacing "." with "dot" and ","
        with "_". It also calls _drop_null_columns to drop columns with every value null

        Args:
            features_df (DataFrame): DataFrame of features on which to perform the
                cleaning.

        Returns:
            DataFrame: Cleaned DataFrame.
        """
        # Renames column names to remove ".", replacing it with "dot" and "," with "_"
        new_cols = [
            F.col(f"`{c}`").alias(c.replace(".", "dot").replace(",", "_"))
            for c in features_df.columns
        ]
        features_df = features_df.select(new_cols)

        # Drops feature column with null values
        features_df = self._drop_null_columns(features_df)

        return features_df

    def _process(self, df: DataFrame) -> DataFrame:
        """Process the DataFrame, calling the pipeline made of:
            - Stacking the input DataFrame (from wide to stacked form)
            - Genereting features from the stacked DataFrame
            - Cleaning the resulting DataFrame of generated features

        Args:
            df (DataFrame): DataFrame with numerical variables from which to perform
                feature generation. Every other column a part from the
                identifier_col_name and time_col_name column are considered numerical
                variables to be used for feature generation.

        Returns:
            DataFrame: DataFrame with a row for every user and a column for every
                feature for every numerical variable present in the input DataFrame.
        """
        stacked_df = self._stack_df(df)
        features_df = self._generate_features(stacked_df)
        cleaned_features_df = self._clean_features_df(features_df)

        return cleaned_features_df

    def __call__(self, df: DataFrame) -> DataFrame:
        self._preprocess(df)
        return self._process(df)
