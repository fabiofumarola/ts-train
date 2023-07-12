from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from .core import AbstractPipelineStep
from typing import Any, List, NewType, Union, Dict


class Filling(AbstractPipelineStep):
    """ """

    def __init__(
        self,
        time_bucket_col_name: str,
        identifier_cols_name: List[str],
        time_bucket_size: int,
        time_bucket_granularity: str,
    ) -> None:
        pass

    """



    come si chiama lo step
    cosa fa lo step?
    Che dati richiede?
    Che tipo di prerequisiti deve soddisfare per essere eseguito?
    Cosa ritorna?
    Cosa facciamo con quello che ritorna?

    """

    def _preprocess(self, df: DataFrame) -> bool:
        # CODICE DA ESEGUIRE ONLINE

        # PYTEST:
        # Fornire diversi input e vericare che tornino true o false
        return True

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        return df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        return result
