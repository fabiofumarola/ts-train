from typing import *

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DataTypeSingleton, DateType, TimestampType
from pyspark.sql.types import StringType, BooleanType
from pyspark.sql.types import (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
)

from ts_train.common.enums import (
    GenericOperator,
    NumericalOperator,
    CategoricalOperator,
    AggFunction,
)


def is_column_present(df: DataFrame, col_name: str) -> bool:
    """Checks if a column is present in the provided DataFrame.

    Args:
        df (DataFrame): DataFrame in which to check the existence.
        col_name (str): Column name of the column to check.

    Returns:
        bool: True if the column is present, False if not
    """
    return col_name in df.columns


def is_dataframe_empty(df: DataFrame) -> bool:
    """Checks if the provided DataFrame is empty.

    Args:
        df (DataFrame): DataFrame on which to perform the check.

    Returns:
        bool: True if the DataFrame is empty, False if not.
    """
    return df.count() == 0


def check_column_dtype(
    df: DataFrame, col_name: str, valid_dtypes: List[DataTypeSingleton]
) -> bool:
    """Checks if the column is of valid type.

    Args:
        df (DataFrame): DataFrame on which to perform the check.
        col_name (str): Column name on which to perform the check.
        valid_dtypes (List[DataType]): DataTypes to be allowed.

    Raises:
        ValueError: with error "Column {col_name} is not a column" if col_name is
            not a column of the DataFrame.

    Returns:
        bool: True if the column is of the right DataType, False if not.
    """
    # Checks if the column is present or not
    if not is_column_present(df, col_name):
        raise ValueError(f"Column {col_name} is not a column")

    col_dtype = df.schema[col_name].dataType

    return col_dtype.__class__ in valid_dtypes


def is_column_categorical(df: DataFrame, col_name: str) -> bool:
    """Checks if the column is categorical (string or boolean).

    Args:
        df (DataFrame): DataFrame on which to perform the check.
        col_name (str): Column name on which to perform the check.

    Returns:
        bool: True if the column is categorical, False if not.
    """
    return check_column_dtype(df, col_name, [StringType, BooleanType])


def is_column_numerical(df: DataFrame, col_name: str) -> bool:
    """Checks if the column contain numeric values (int, float, double...).

    Args:
        df (DataFrame): DataFrame on which to perform the check.
        col_name (str): Column name on which to perform the check.

    Returns:
        bool: True if the column is numerical, False if not.
    """
    # Define a list of numeric data types
    all_numeric_types: List[DataTypeSingleton] = [
        ByteType,
        ShortType,
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
    ]

    return check_column_dtype(df, col_name, all_numeric_types)


def is_column_timestamp(df: DataFrame, col_name: str) -> bool:
    """Checks if the column is a date or timestamp.

    Args:
        df (DataFrame): DataFrame on which to perform the check.
        col_name (str): Column name on which to perform the check.

    Returns:
        bool: True if the column is a timestamp, False if not.
    """
    return check_column_dtype(df, col_name, [DateType, TimestampType])


def is_column_window(df: DataFrame, col_name: str) -> bool:
    """Checks if the column is a window.

    Args:
        df (DataFrame): DataFrame on which to perform the check.
        col_name (str): Column name on which to perform the check.

    Raises:
        ValueError: with error "Column {col_name} is not a column" if col_name is
            not a column of the DataFrame.

    Returns:
        bool: True if the column is a window, False if not.
    """
    # Checks if the column is present or not
    if not is_column_present(df, col_name):
        raise ValueError(f"Column {col_name} is not a column")

    return (
        str(df.schema[col_name].dataType)
        == "StructType([StructField('start', TimestampType(), True),"
        " StructField('end', TimestampType(), True)])"
    )


# Fixtures helper functions
def cast_column_to_timestamp(
    df: DataFrame, col_name: str, format: str = "yyyy-MM-dd"
) -> DataFrame:
    return df.withColumn(col_name, F.to_timestamp(F.col(col_name), format))


def cast_columns_to_timestamp(
    df: DataFrame, cols_name: list[str], format: str = "yyyy-MM-dd"
) -> DataFrame:
    for col_name in cols_name:
        df = cast_column_to_timestamp(df, col_name, format)
    return df


def cast_struct_to_timestamps(
    df: DataFrame,
    struct_col_name: str,
    struct_fields_name: Tuple[str, str] = ("start", "end"),
    format: str = "yyyy-MM-dd",
) -> DataFrame:
    struct_field_start = f"{struct_col_name}.{struct_fields_name[0]}"
    struct_field_end = f"{struct_col_name}.{struct_fields_name[1]}"

    return df.withColumn(
        struct_col_name,
        F.struct(
            F.to_timestamp(F.col(struct_field_start), format).alias(
                struct_fields_name[0]
            ),
            F.to_timestamp(F.col(struct_field_end), format).alias(
                struct_fields_name[1]
            ),
        ),
    )


def create_timestamps_struct(
    df: DataFrame,
    cols_name: Tuple[str, str],
    struct_col_name: str,
    struct_fields_name: Tuple[str, str] = ("start", "end"),
    format: str = "yyyy-MM-dd",
) -> DataFrame:
    return df.withColumn(
        struct_col_name,
        F.struct(
            F.to_timestamp(F.col(cols_name[0]), format).alias(struct_fields_name[0]),
            F.to_timestamp(F.col(cols_name[1]), format).alias(struct_fields_name[1]),
        ),
    )  # .drop(*cols_name)


def parse_operator(
    operator_str: str,
) -> Union[GenericOperator, NumericalOperator, CategoricalOperator]:
    for operator in [*GenericOperator, *NumericalOperator, *CategoricalOperator]:
        if operator_str == operator.value:
            return operator
    raise ValueError("Operator not allowed")


def parse_agg_function(agg_function_str: str) -> AggFunction:
    for agg_function in AggFunction:
        if agg_function_str == agg_function.value:
            return agg_function
    raise ValueError("Aggregation function not allowed")
