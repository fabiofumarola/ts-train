from typing import *

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DataTypeSingleton, DateType, TimestampType


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
