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

from ts_train.step.core import AbstractPipelineStep
from ts_train.common.utils import *
from ts_train.common.enums import (
    NumericalOperator,
    CategoricalOperator,
    GenericOperator,
)


@dataclass
class Filter:
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
    filters: List[Union[Filter, AndGroup, OrGroup]]
    name: Union[str, None] = None


@dataclass
class OrGroup:
    filters: List[Union[Filter, AndGroup, OrGroup]]
    name: Union[str, None] = None


@dataclass
class Pivot:
    col_name: StrictStr
    operator: Annotated[
        StrictStr,
        AfterValidator(lambda v: parse_operator(v)),
    ]
    # TODO eliminare la list[Any] che consente l'utilizzo di []
    value: Union[list[StrictStr], list[StrictBool], list[StrictInt], None]
    name: Union[str, None] = None


class Aggregation(BaseModel):
    numerical_col_name: StrictStr
    agg_function: Annotated[StrictStr, AfterValidator(lambda v: parse_agg_function(v))]
    filters: Union[list[Union[Filter, AndGroup, OrGroup]], None] = None
    pivot: Union[Pivot, None] = None
    new_col_name: Union[StrictStr, None] = None


class Aggregating(AbstractPipelineStep, BaseModel):
    identifier_cols_name: conlist(StrictStr, min_length=1)  # type: ignore
    time_bucket_cols_name: conlist(StrictStr, min_length=1)  # type: ignore
    aggregations: conlist(Aggregation, min_length=1)  # type: ignore

    def _preprocess(self, df: DataFrame) -> None:
        # Checks if the DataFrame is full or empty
        if is_dataframe_empty(df):
            raise ValueError("Empty DataFrame")

        # Checks if identifier_cols_name are columns
        for identifier_col_name in self.identifier_cols_name:
            if not is_column_present(df, identifier_col_name):
                raise ValueError(f"Column {identifier_col_name} is not a column")

        # Checks if time_bucket_col_name is a column and is a timebucket
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
        for filter in filters:
            if isinstance(filter, (AndGroup, OrGroup)):
                # It's a AndGroup or OrGroup
                self._check_filters(df, filter.filters)
            else:
                # It's a Filter
                self._check_filter(df, filter)

    def _check_filter(self, df: DataFrame, filter: Filter) -> None:
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
        if pivot.value is None:
            pivot.value = [
                row[0] for row in df.select(pivot.col_name).distinct().collect()
            ]

        return pivot

    def _parse_value(self, value) -> Any:
        return value

    def _compute_filter_condition(self, filter: Filter) -> Column:
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
        return df.groupBy(extended_identifier_cols_name)

    def _pivot(
        self,
        df: GroupedData,
        pivot: Union[Pivot, None],
    ) -> GroupedData:
        # Returns original DataFrame if no pivot filter is present
        if pivot is None:
            return df

        # Pivotes
        pivoted_df: GroupedData = df.pivot(pivot.col_name, pivot.value)  # type: ignore

        return pivoted_df

    def _aggregate(self, df: GroupedData, aggregation: Aggregation) -> DataFrame:
        return df.agg({aggregation.numerical_col_name: str(aggregation.agg_function)})

    def _generate_filter_name(self, filter: Filter) -> str:
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
        if aggregation.pivot is not None and aggregation.pivot.value is not None:
            return [str(value) for value in aggregation.pivot.value]
        else:
            return [f"{aggregation.agg_function}({aggregation.numerical_col_name})"]

    def simulate_renaming(self, aggregations: list[Aggregation]) -> None:
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
        old_cols_name = self._generate_old_cols_name(aggregation)
        new_cols_name = self._generate_new_cols_name(
            aggregation, aggregation.new_col_name
        )

        for old_col_name, new_col_name in zip(old_cols_name, new_cols_name):
            df = df.withColumnRenamed(old_col_name, new_col_name)

        return df

    def _join_dataframes(self, df1: DataFrame, df2: DataFrame) -> DataFrame:
        join_columns = [col_name for col_name in df1.columns if col_name in df2.columns]

        return df1.join(df2, join_columns)

    def _aggregation_pipeline(
        self,
        df: DataFrame,
        aggregation: Aggregation,
        extended_identifier_cols_name: list[str],
    ) -> DataFrame:
        filtered_df = self._filter(df, aggregation.filters)
        grouped_df = self._groupby(filtered_df, extended_identifier_cols_name)
        pivoted_df = self._pivot(grouped_df, aggregation.pivot)
        aggregated_df = self._aggregate(pivoted_df, aggregation)
        renamed_df = self._rename(aggregated_df, aggregation)

        return renamed_df

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
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
        return result
