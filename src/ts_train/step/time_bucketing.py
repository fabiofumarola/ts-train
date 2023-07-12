from typing import *
from enum import Enum

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pydantic import BaseModel, StrictInt, StrictStr

from ts_train.step.core import AbstractPipelineStep


class TimeBucketGranularity(str, Enum):
    WEEK = "week"
    WEEKS = "weeks"
    DAY = "day"
    DAYS = "days"
    HOUR = "hour"
    HOURS = "hours"
    MINUTE = "minute"
    MINUTES = "minutes"


class TimeBucketing(AbstractPipelineStep, BaseModel):
    """
    Associate each row with a interval of time (bucket)

    It create a new colomn named bucket that is composed by a stard and end timestamp
    """

    time_column_name: StrictStr
    time_bucket_size: StrictInt
    time_bucket_granularity: TimeBucketGranularity
    time_bucket_col_name: StrictStr

    def __init__(
        self,
        **kargs,
    ) -> None:
        # TODO remove the init if no other operation is done a part from super()
        super().__init__(**kargs)

    """
    come si chiama lo step
    cosa fa lo step?
    Che dati richiede?
    Che tipo di prerequisiti deve soddisfare per essere eseguito?
    Cosa ritorna?
    Cosa facciamo con quello che ritorna?

    TEST DA EFFETTUARE

    """

    # @classmethod
    # def from_config(cls: TimeBucketing, config: Any) -> TimeBucketing:
    #    raise NotImplementedError

    def _preprocess(self, df: DataFrame) -> None:
        # CODICE DA ESEGUIRE
        # check if df is not empty
        # check if there is at least a timestamp column
        # check if windows size is a valid string
        # check if column_name is a valid string
        # check if column name is in df schema

        # Checks if the DataFrame is full or empty
        if df.count() == 0:
            raise ValueError("Empty DataFrame")

        # Checks if time_column_name is a column and is a timestamp column
        def get_dtype(col_name):
            for name, dtype in df.dtypes:
                if name == col_name:
                    return dtype
            return None

        if self.time_column_name not in df.columns:
            raise ValueError(f"Column {self.time_column_name} is not a column")

        if get_dtype(self.time_column_name) not in ["timestamp", "date", "datetime"]:
            raise ValueError(f"Column {self.time_column_name} not a timestamp column")

        # Column name arriva fino alla settimana. Aggiugnere mese/anno
        # Nelle config esplicitare la lista dei column name che si possono usare

        # testare la presenza o meno di eccezioni
        return None

    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        # Creare la colonna con i timebucket e aggiungerla al df

        # TEST:
        # ASSERT len(new_df.columns) == len(df.columns)  + 1
        # timestsamp sia all' interno della windows creata
        return df

    def _postprocess(self, result: DataFrame) -> DataFrame:
        return result
