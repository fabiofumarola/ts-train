from __future__ import annotations
from typing import *
from functools import reduce

from pyspark.sql import SparkSession, Column, GroupedData
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    StringType,
    BooleanType,
)

from pydantic import (
    BaseModel,
    conlist,
    StrictStr,
    StrictBool,
    StrictInt,
    StrictFloat,
    model_validator,
    AfterValidator,
)
from pydantic.dataclasses import dataclass

from ts_train.transactions_to_timeseries.core import AbstractPipelineStep
from ts_train.common.utils import *
from ts_train.common.enums import (
    NumericalOperator,
    CategoricalOperator,
    GenericOperator,
)


@dataclass
class Filter:
    """Used to filter an Aggregation. This will filter out some input sample to the
    aggregation focucing the resulting aggregated column.

    Attributes:
        col_name (StrictStr): Name of the column on which to perform the filter.
        operator (StrictStr): Operator to be used for the filter. It can be one of the
            followings:
                - GenericOperator: =, ==, !=
                - NumericalOperator: <, <=, >, >=
                - CategoricalOperator: in, notin, not in
        value (Union[Union[list[StrictStr], list[StrictBool], list[StrictInt]],
            StrictStr, StrictInt, StrictFloat, StrictBool]): value used for the filter
            operation. It should be consistent with the choosen operator.
        name (str, Optional): name for the specific filter. This could be used by groups
            containing this Filter or by the Aggregation to choose for the new column/s
            name/s.

    Raises:
        ValueError: with message "Value ({value}) not allowed for numerical operator
            ({operator})" when operator and value are not consistent.
        ValueError: with message "Value ({value}) not allowed for categorical operator
            ({operator})" when operator and value are not consistent.
        ValueError: with message "Value ({value}) not allowed for generic operator
            ({operator})" when operator and value are not consistent.
    """

    col_name: StrictStr
    operator: Annotated[
        StrictStr,
        AfterValidator(lambda v: parse_operator(v)),
    ]
    value: Union[
        Union[list[StrictStr], list[StrictBool], list[StrictInt]],
        StrictStr,
        StrictInt,
        StrictFloat,
        StrictBool,
    ]
    name: Union[str, None] = None

    @model_validator(mode="after")  # type: ignore
    def check_consistency_operator_value(self) -> "Filter":
        if type(self.operator) == NumericalOperator and type(self.value) not in [
            int,
            float,
        ]:
            raise ValueError(
                f"Value ({self.value}) not allowed for numerical operator"
                f" ({self.operator})"
            )
        elif type(self.operator) == CategoricalOperator and type(self.value) not in [
            list
        ]:
            raise ValueError(
                f"Value ({self.value}) not allowed for categorical operator"
                f" ({self.operator})"
            )
        elif type(self.operator) == GenericOperator and type(self.value) in [list]:
            raise ValueError(
                f"Value ({self.value}) not allowed for generic operator"
                f" ({self.operator})"
            )

        return self


@dataclass
class AndGroup:
    """Used to combine multiple Filter operators (or other AndGroup/OrGroup) with an and
    logic.

    Attributes:
        filters (List[Union[Filter, AndGroup, OrGroup]]): Filter/AndGroup/OrGroup to be
            concatenated in and condition.
        name (str, Optional): name for the specific group. This could be used by groups
            containing this group or by the Aggregation to choose for the new column/s
            name/s.
    """

    filters: List[Union[Filter, AndGroup, OrGroup]]
    name: Union[str, None] = None


@dataclass
class OrGroup:
    """Used to combine multiple Filter operators (or other AndGroup/OrGroup) with a or
    logic.

    Attributes:
        filters (List[Union[Filter, AndGroup, OrGroup]]): Filter/AndGroup/OrGroup to be
            concatenated in or condition.
        name (str, Optional): name for the specific group. This could be used by groups
            containing this group or by the Aggregation to choose for the new column/s
            name/s.
    """

    filters: List[Union[Filter, AndGroup, OrGroup]]
    name: Union[str, None] = None


@dataclass
class Pivot:
    """Used to apply a pivot operation an Aggregation. This will generate more than one
    column for each Aggregation in which it is used. The number of generated columns
    depends on the number of options of the column selected for the pivot operation.

    Attributes:
        col_name (StrictStr): Name of the column on which to perform the pivot.
        operator (StrictStr): Operator to be used for the pivot. It can be one of the
            followings:
                - CategoricalOperator: in, notin, not in
        value (Union[list[StrictStr], list[StrictBool], list[StrictInt], None]): list
            of options to be used for the pivot. If None or an empty list are used every
            option in the column will be used in the pivot operation. If you provide a
            list of options only those will be used.
        name (str, Optional): name for the specific pivot. This could be used by groups
            containing this Filter or by the Aggregation to choose for the new column/s
            name/s. You can use the special token "PIVOTVALUE" that will be replaced by
            the specific option used in that column name.
    """

    col_name: StrictStr
    operator: Annotated[
        StrictStr,
        AfterValidator(lambda v: parse_operator(v)),
    ]
    value: Union[list[StrictStr], list[StrictBool], list[StrictInt], None]
    name: Union[str, None] = None


class Aggregation(BaseModel):
    """Aggregation is the core concept of the Aggregating step. It represents a new
    column in the DataFrame containing aggregated values. Pay attention that, if you use
    a Pivot operation, you will end up with more than one column.

    Attributes:
        numerical_col_name (StrictStr): name of the numerical column on which to perform
            the aggregation. These are the numbers to be aggregated.
        agg_function (StrictStr): aggregation function to be used for the numerical
            aggregation. Current supported operations are: sum, count, avg, min, max,
            first, last.
        filters (list[Union[Filter, AndGroup, OrGroup], Optional): list of Filter
            operation or groups (AndGroup or OrGroup). If you provide at least one a
            filtering of rows will be performed, if None or empty list every sample will
            be used for the aggregation.
        pivot (Pivot, Optional): pivot operation. If not provided no pivot operation
            will be used.
        new_col_name (StrictStr, Optional): If not provided (None) the new column name
            (or columns names if pivot is used) will be automatically generated. If you
            provide this param this will be used. In this string you can use special
            tokens like:
                - FILTERS: this will be replaced with the provided or automatic name of
                    the filters.
                - FUNCTION: this will be replaced with the name of the aggregation
                    function used.
                - NUMERICAL: this will be replace with the column name of the numerical
                    column.
                - PIVOT: this will be replaced with the provided or automatic name of
                    the pivot operation.
                - PIVOTVALUE: this will be replaced with the option of the column on
                    which the pivot operation has been done.
    """

    numerical_col_name: StrictStr
    agg_function: Annotated[StrictStr, AfterValidator(lambda v: parse_agg_function(v))]
    filters: Union[list[Union[Filter, AndGroup, OrGroup]], None] = None
    pivot: Union[Pivot, None] = None
    new_col_name: Union[StrictStr, None] = None


class Aggregating(AbstractPipelineStep, BaseModel):
    """Aggregating takes a DataFrame in input and performs aggregations. This will take
    samples and will group them in the same time bucket performing aggregation functions

    Attributes:
        identifier_cols_name (list[StrictStr]): list of names of columns used to
            identify a used from another.
        time_bucket_cols_name (list[StrictStr]): list of names of columns used to group
            samples in the same time bucket.
        aggregations (list[Aggregation]): list of Aggregation to be performed. You have
            to provide at least one Aggregation.
    """

    identifier_cols_name: conlist(StrictStr, min_length=1)  # type: ignore
    time_bucket_cols_name: conlist(StrictStr, min_length=1)  # type: ignore
    aggregations: conlist(Aggregation, min_length=1)  # type: ignore

    def _preprocess(self, df: DataFrame) -> None:
        """Performs checks on the DataFrame, on other parameters and on the provided
        aggregations.

        Args:
            df (DataFrame): DataFrame on which to perform the check.

        Raises:
            ValueError: with message "Empty DataFrame" if the provided DataFrame is
                empty.
            ValueError: with message "Column {identifier_col_name} is not a column" if
                one of the identifier_cols_name is not present in the DataFrame.
            ValueError: with message "Column {time_bucket_col_name} is not a column" if
                one of the time_bucket_cols_name is not present in the DataFrame.
            ValueError: with message "Column {time_bucket_col_name} is not a timestamp"
                if the time_bucket_col_name is not a time column.
        """
        # Checks if the DataFrame is full or empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        # Checks if identifier_cols_name are columns
        for identifier_col_name in self.identifier_cols_name:
            if not is_column_present(df, identifier_col_name):
                raise ValueError(f"Column {identifier_col_name} is not a column")

        # Checks if time_bucket_col_name is a column and is a timestamp
        for time_bucket_col_name in self.time_bucket_cols_name:
            if not is_column_present(df, time_bucket_col_name):
                raise ValueError(f"Column {time_bucket_col_name} is not a column")

            if not is_column_timestamp(df, time_bucket_col_name):
                raise ValueError(f"Column {time_bucket_col_name} is not a timestamp")

        # Checks and preprocesses aggregations
        self.aggregations: List[Aggregation]
        for aggregation in self.aggregations:
            # Checks and preprocesses aggregation attributes validity
            self._check_aggregation(df, aggregation)

    def _check_aggregation(self, df: DataFrame, aggregation: Aggregation) -> None:
        """Checks that the provided Aggregation is a good one for the DataFrame provided

        Args:
            df (DataFrame): DataFrame on which to perform the test.
            aggregation (Aggregation): Aggregation to check.

        Raises:
            ValueError: with message "Column {numerical_col_name} is not a column" if
                the numerical_col_name is not present in the provided DataFrame.
            ValueError: with message "Column {numerical_col_name} is not a numeric
                column"
            ValueError: with message "Column {new_col_name} is already a column" if the
                new column is already present. If a special TOKEN is provided the check
                does not guarantee to work.
        """
        # Allows to input empty list instead of None for filters
        if isinstance(aggregation.filters, list) and len(aggregation.filters) == 0:
            aggregation.filters = None

        # Allows to input empty list instead of None for filters
        if isinstance(aggregation.filters, list) and len(aggregation.filters) == 0:
            aggregation.filters = None

        # Checks existency and type on numerical column
        if not is_column_present(df, aggregation.numerical_col_name):
            raise ValueError(f"Column {aggregation.numerical_col_name} is not a column")

        if not is_column_numerical(df, aggregation.numerical_col_name):
            raise ValueError(
                f"Column {aggregation.numerical_col_name} is not a numeric column"
            )

        # Checks validity of filters and preprocess
        if aggregation.filters is not None:
            self._check_filters(df, aggregation.filters)

        # Preprocess Pivot
        if aggregation.pivot is not None:
            pivot: Pivot = aggregation.pivot
            aggregation.pivot = self._preprocess_pivot(df, pivot)

        # Checks new col name
        if aggregation.new_col_name is not None:
            if is_column_present(df, aggregation.new_col_name):
                raise ValueError(
                    f"Column {aggregation.new_col_name} is already a column"
                )

    def _check_filters(
        self, df: DataFrame, filters: list[Union[Filter, AndGroup, OrGroup]]
    ) -> None:
        """Checks filter (or group of them) provided to the Aggregation.

        Args:
            df (DataFrame): DataFrame on which to perform the check.
            filters (list[Union[Filter, AndGroup, OrGroup]]): Filter or groups to check.
        """
        for filter in filters:
            if isinstance(filter, (AndGroup, OrGroup)):
                # It's a AndGroup or OrGroup
                self._check_filters(df, filter.filters)
            else:
                # It's a Filter
                self._check_filter(df, filter)

    def _check_filter(self, df: DataFrame, filter: Filter) -> None:
        """Checks a single filter.

        Args:
            df (DataFrame): DataFrame on which to perform the check.
            filter (Filter): Filter to check.

        Raises:
            ValueError: with message "Column {col_name} is not a column" if col_name is
                not a column of the DataFrame.
            ValueError: with message "Mismatch types between {col_name} column
                ({col_dtype}) and {value} value ({value_dtype})" if the tpye of value of
                the filter does not match the type of the provided column on which to
                perform the filter operation.
        """
        # Checks existency and type on categorical column
        if not is_column_present(df, filter.col_name):
            raise ValueError(f"Column {filter.col_name} is not a column")

        # Checks consistency of column type and value
        col_dtype = type(df.schema[filter.col_name].dataType)
        value_dtype = type(filter.value)
        allowed_value_and_col_dtypes: dict[object, list[Any]] = {
            int: [ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType],
            float: [FloatType, DoubleType],
            str: [StringType],
            bool: [BooleanType],
            list: [
                ByteType,
                ShortType,
                IntegerType,
                LongType,
                FloatType,
                DoubleType,
                StringType,
                BooleanType,
            ],
        }

        allowed_col_dtypes = allowed_value_and_col_dtypes[value_dtype]
        if col_dtype not in allowed_col_dtypes:
            raise ValueError(
                f"Mismatch types between {filter.col_name} column ({col_dtype}) and"
                f" {filter.value} value ({value_dtype})"
            )

    def _preprocess_pivot(self, df: DataFrame, pivot: Pivot) -> Pivot:
        """Preprocess a Pivot operation. If a None or an empty list is provided as value
        this method will put inside every possible options contained in the column.

        Args:
            df (DataFrame): DataFrame on which to find the column.
            pivot (Pivot): Pivot operation to preprocess.

        Returns:
            Pivot: Preprocessed Pivot operation with values provided.
        """
        if pivot.value is None or (
            isinstance(pivot.value, list) and len(pivot.value) == 0
        ):
            pivot.value = [
                row[0] for row in df.select(pivot.col_name).distinct().collect()
            ]

        return pivot

    def _parse_value(self, value) -> Any:
        return value

    def _compute_filter_condition(self, filter: Filter) -> Column:
        """Computes a filter condition. Now it is ready to be concatenated with other
        filter operation and be used to filter the DataFrame.

        Args:
            filter (Filter): Filter operation to be computed

        Raises:
            ValueError: with message "Filter with operator not allowed" if a not allowed
                operator is provided.

        Returns:
            Column: Condition of the Filter operation.
        """
        if filter.operator == "in":
            return F.col(filter.col_name).isin(filter.value)
        elif filter.operator in ["not in", "notin"] and isinstance(filter.value, list):
            return ~F.col(filter.col_name).isin(filter.value)
        elif filter.operator in ["=", "=="]:
            return F.col(filter.col_name) == self._parse_value(filter.value)
        elif filter.operator == "!=":
            return F.col(filter.col_name) != filter.value
        elif filter.operator == "<":
            return F.col(filter.col_name) < self._parse_value(filter.value)
        elif filter.operator == "<=":
            return F.col(filter.col_name) <= self._parse_value(filter.value)
        elif filter.operator == ">":
            return F.col(filter.col_name) > self._parse_value(filter.value)
        elif filter.operator == ">=":
            return F.col(filter.col_name) >= self._parse_value(filter.value)
        else:
            raise ValueError("Filter with operator not allowed")

    def _compute_filters_expression(
        self, filters: list[Union[Filter, AndGroup, OrGroup]], mode="and"
    ) -> Column:
        """Concatenates the filters or the groups into one filter expression to be used
        to filter samples of the DataFrame.

        Args:
            filters (list[Union[Filter, AndGroup, OrGroup]]): filters or groups of them
                to be used to compute the full expression.
            mode (str, optional): "and" or "or" mode. Defaults to "and".

        Returns:
            Column: Combined filter contidiont to create a filter expression.
        """
        filters_expressions = []
        for filter in filters:
            if isinstance(filter, AndGroup):
                filters_expressions.append(
                    self._compute_filters_expression(filter.filters, mode="and")
                )
            elif isinstance(filter, OrGroup):
                filters_expressions.append(
                    self._compute_filters_expression(filter.filters, mode="or")
                )
            else:
                filters_expressions.append(self._compute_filter_condition(filter))

        if mode == "and":
            filters_expression = reduce(lambda f1, f2: f1 & f2, filters_expressions)
        else:
            filters_expression = reduce(lambda f1, f2: f1 | f2, filters_expressions)

        return filters_expression

    def _filter(
        self,
        df: DataFrame,
        filters: Union[list[Union[Filter, AndGroup, OrGroup]], None],
    ) -> DataFrame:
        """Executes the filter operations if filters are provided to the Aggregation.

        Args:
            df (DataFrame): DataFrame on which to perform the filter operation.
            filters (Union[list[Union[Filter, AndGroup, OrGroup]], None]): filters to be
                used during the filtering operation.

        Returns:
            DataFrame: DataFrame filtered.
        """
        # Returns the original DataFrame if no select filter is present
        if filters is None:
            return df

        # Creates filter expression
        filter_expression = self._compute_filters_expression(filters)

        # Filters the DataFrame
        filtered_df = df.filter(filter_expression)

        # return filtered_df

        # Creates a DataFrame with every distinct user and time bucket
        distinct_df = df.select(
            *self.identifier_cols_name, *self.time_bucket_cols_name
        ).distinct()

        # Joins the distinct_df to add null values for missing rows
        filtered_df = filtered_df.join(
            distinct_df,
            on=[*self.identifier_cols_name, *self.time_bucket_cols_name],
            how="outer",
        )

        return filtered_df

    def _groupby(
        self, df: DataFrame, extended_identifier_cols_name: list[str]
    ) -> GroupedData:
        """Groups the DataFrame with the provided identifier columns.

        Args:
            df (DataFrame): DataFrame to group
            extended_identifier_cols_name (list[str]): list of name of columns used to
                group samples.

        Returns:
            GroupedData: grouped DataFrame in GroupedData form.
        """
        return df.groupBy(extended_identifier_cols_name)

    def _pivot(
        self,
        df: GroupedData,
        pivot: Union[Pivot, None],
    ) -> GroupedData:
        """Executes the pivot operation if pivot is provided to the Aggregation.

        Args:
            df (GroupedData): GroupedData on which to perform the pivot operation.
            pivot (Union[Pivot, None]): pivot to be used during the pivot operation.

        Returns:
            GroupedData: pivoted DataFrame in GroupedData form.
        """
        # Returns original DataFrame if no pivot filter is present
        if pivot is None:
            return df

        # Pivotes
        pivoted_df: GroupedData = df.pivot(pivot.col_name, pivot.value)  # type: ignore

        return pivoted_df

    def _aggregate(self, df: GroupedData, aggregation: Aggregation) -> DataFrame:
        """Aggregates the GroupdData DataFrame with the provided Aggregation.

        Args:
            df (GroupedData): GroupdData DataFrame to be aggregated.
            aggregation (Aggregation): Aggregation to be used to aggregated.

        Returns:
            DataFrame: DataFrame with aggregated new columns.
        """
        return df.agg({aggregation.numerical_col_name: str(aggregation.agg_function)})

    def _generate_filter_name(self, filter: Filter) -> str:
        """Generates Filter name automatic or using the name param.

        Args:
            filter (Filter): Filter for the name to be generated.

        Returns:
            str: Generated name of the Filter.
        """
        if isinstance(filter.operator, CategoricalOperator) and isinstance(
            filter.value, list
        ):
            values = [str(value) for value in filter.value]
            categorical_values = "_".join(values)
            return f"{filter.col_name}[{categorical_values}]"
        else:
            if filter.operator == "==":
                operator = "="
            else:
                operator = filter.operator
            return f"{filter.col_name}{operator}{filter.value}"

    def _generate_filters_name(
        self,
        filters: Optional[List[Union[Filter, AndGroup, OrGroup]]],
        mode="first_and",
    ) -> str:
        """Generates name of filters concatenated using and/or groups.

        Args:
            filters (Optional[List[Union[Filter, AndGroup, OrGroup]]]): filters or group
                to be used to generate a combined name.
            mode (str, optional): If "first_and" is used it means this is the first
                layer if filters. Defaults to "first_and".

        Returns:
            str: Global name for every filter provided.
        """
        if filters is None:
            return ""

        names = []
        for filter in filters:
            name = ""
            if type(filter) == OrGroup:
                if filter.name is not None:
                    name = filter.name
                else:
                    name = self._generate_filters_name(filter.filters, mode="or")
            elif type(filter) == AndGroup:
                if filter.name is not None:
                    name = filter.name
                else:
                    name = self._generate_filters_name(filter.filters, mode="and")
            elif type(filter) == Filter:
                if filter.name is not None:
                    name = filter.name
                else:
                    name = self._generate_filter_name(filter)
            names.append(name)

        if mode == "and":
            return "(" + "&".join(names) + ")"
        elif mode == "or":
            return "(" + "|".join(names) + ")"
        else:
            return "&".join(names)

    def _generate_pivot_names(self, pivot: Pivot) -> list[str]:
        """Generates a name for the Pivot operation.

        Args:
            pivot (Pivot): Pivot operation to be used to generate the name.

        Returns:
            list[str]: Generated name for the Pivot operation.
        """
        if pivot is not None and pivot.value is not None:
            if pivot.name is None:
                return [f"{pivot.col_name}={value}" for value in pivot.value]
            else:
                if "PIVOTVALUE" in pivot.name:
                    prefix, suffix = pivot.name.split("PIVOTVALUE")
                    return [f"{prefix}{value}{suffix}" for value in pivot.value]
                else:
                    return [f"{pivot.name}{value}" for value in pivot.value]
        else:
            return [""]

    def _generate_new_cols_name(
        self, aggregation: Aggregation, pattern: Optional[str] = None
    ) -> list[str]:
        """Generates a names for the specific Aggregation provided.

        Args:
            aggregation (Aggregation): Aggregation to be used.
            pattern (Optional[str], optional): Pattern to be used. Defaults to None.

        Returns:
            list[str]: list of new columns name generated.
        """

        def generate_pattern(pattern: Optional[str]) -> str:
            if pattern is None:
                if aggregation.pivot is None and aggregation.filters is None:
                    return "FUNCTION(NUMERICAL)"
                elif aggregation.pivot is None and aggregation.filters is not None:
                    return "FUNCTION(NUMERICAL)_where_FILTERS"
                elif aggregation.pivot is not None and aggregation.filters is None:
                    return "FUNCTION(NUMERICAL)_where_PIVOT"
                else:
                    return "FUNCTION(NUMERICAL)_where_FILTERS_PIVOT"
            else:
                return pattern

        def parse_pattern_for_function(aggregation: Aggregation, pattern: str) -> str:
            return pattern.replace("FUNCTION", aggregation.agg_function)

        def parse_pattern_for_numerical(aggregation: Aggregation, pattern: str) -> str:
            return pattern.replace("NUMERICAL", aggregation.numerical_col_name)

        def parse_pattern_for_filters(aggregation: Aggregation, pattern: str) -> str:
            return pattern.replace(
                "FILTERS", self._generate_filters_name(aggregation.filters)
            )

        def parse_pattern_for_pivot(
            aggregation: Aggregation, pattern: str
        ) -> list[str]:
            if aggregation.pivot is None:
                return [pattern]
            else:
                if "PIVOT" in pattern:
                    return [
                        pattern.replace("PIVOT", name)
                        for name in self._generate_pivot_names(aggregation.pivot)
                    ]
                else:
                    return [
                        f"{pattern}_{name}"
                        for name in self._generate_pivot_names(aggregation.pivot)
                    ]

        def parse_pattern_for_pivotvalue(
            aggregation: Aggregation, pattern: str
        ) -> list[str]:
            if aggregation.pivot is None or aggregation.pivot.value is None:
                return [pattern]
            else:
                return [
                    pattern.replace("PIVOTVALUE", str(value))
                    for value in aggregation.pivot.value
                ]

        pattern = generate_pattern(pattern)

        pattern = parse_pattern_for_function(aggregation, pattern)
        pattern = parse_pattern_for_numerical(aggregation, pattern)
        pattern = parse_pattern_for_filters(aggregation, pattern)

        if "PIVOTVALUE" in pattern:
            new_cols_name = parse_pattern_for_pivotvalue(aggregation, pattern)
        else:
            new_cols_name = parse_pattern_for_pivot(aggregation, pattern)

        return new_cols_name

    def _generate_old_cols_name(self, aggregation: Aggregation) -> list[str]:
        """Generates old columns name. This is used for the replacement of column names.

        Args:
            aggregation (Aggregation): Aggregation for the old names to be generated.

        Returns:
            list[str]: List of old names.
        """
        if aggregation.pivot is not None and aggregation.pivot.value is not None:
            return [str(value) for value in aggregation.pivot.value]
        else:
            return [f"{aggregation.agg_function}({aggregation.numerical_col_name})"]

    def simulate_renaming(self, aggregations: list[Aggregation]) -> None:
        """Prints the old names of columns for the list of Aggregation provided and the
        new names. This is helful if you want to simulate the columns names for the new
        DataFrame.

        Args:
            aggregations (list[Aggregation]): list of Aggregation desired.
        """
        for aggregation in aggregations:
            old_cols_name = self._generate_old_cols_name(aggregation)
            new_cols_name = self._generate_new_cols_name(
                aggregation, aggregation.new_col_name
            )

            print(f"Aggregation: {aggregation}")
            for old_col_name, new_col_name in zip(old_cols_name, new_cols_name):
                print(f"{old_col_name} -> {new_col_name}")
            print()

    def _rename(self, df: DataFrame, aggregation: Aggregation) -> DataFrame:
        """Renames the DataFrame for a specific Aggregation.

        Args:
            df (DataFrame): DataFrame to be used for the renaming.
            aggregation (Aggregation): Aggregation to be used for the renaming.

        Returns:
            DataFrame: Renamed DataFrame for the single Aggregation provided
        """
        old_cols_name = self._generate_old_cols_name(aggregation)
        new_cols_name = self._generate_new_cols_name(
            aggregation, aggregation.new_col_name
        )

        for old_col_name, new_col_name in zip(old_cols_name, new_cols_name):
            df = df.withColumnRenamed(old_col_name, new_col_name)

        return df

    def _join_dataframes(self, df1: DataFrame, df2: DataFrame) -> DataFrame:
        """Joins two DataFrames mamaging duplicated columns.

        Args:
            df1 (DataFrame): First DataFrame.
            df2 (DataFrame): Second DataFrame.

        Returns:
            DataFrame: Joined DataFrame.
        """
        join_columns = [col_name for col_name in df1.columns if col_name in df2.columns]

        return df1.join(df2, join_columns)

    def _aggregation_pipeline(
        self,
        df: DataFrame,
        aggregation: Aggregation,
        extended_identifier_cols_name: list[str],
    ) -> DataFrame:
        """Full pipeline for the Aggregating step. It filters, groups, pivots,
        aggregates and renames.

        Args:
            df (DataFrame): DataFrame on which to perform the Aggregating step.
            aggregation (Aggregation): Aggregation to perform.
            extended_identifier_cols_name (list[str]): List of identifier column names.

        Returns:
            DataFrame: Resulting DataFrame for the specified Aggregation
        """
        filtered_df = self._filter(df, aggregation.filters)
        grouped_df = self._groupby(filtered_df, extended_identifier_cols_name)
        pivoted_df = self._pivot(grouped_df, aggregation.pivot)
        aggregated_df = self._aggregate(pivoted_df, aggregation)
        renamed_df = self._rename(aggregated_df, aggregation)

        return renamed_df

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """Executes the full Aggregating pipeline for every Aggregation.

        Args:
            df (DataFrame): DataFrame on which to perform every Aggregation requested.
            spark (SparkSession): Spark session to be used.

        Returns:
            DataFrame: Resulting DataFrame with every new column added.
        """
        extended_identifier_cols_name: list[str] = [
            *self.identifier_cols_name,
            *self.time_bucket_cols_name,
        ]

        aggregated_dfs = []
        for aggregation in self.aggregations:
            aggregated_df = self._aggregation_pipeline(
                df, aggregation, extended_identifier_cols_name
            )
            aggregated_dfs.append(aggregated_df)

        result_df = reduce(
            lambda df1, df2: self._join_dataframes(df1, df2),
            aggregated_dfs,
        )

        result_df = result_df.orderBy(*extended_identifier_cols_name)

        return result_df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        """Postprocess step, in this case with not meaning.

        Args:
            result (DataFrame): DataFrame from the _process step.

        Returns:
            DataFrame: DataFrame as the input one as the step is skipped.
        """
        return result
