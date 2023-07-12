from abc import ABC, abstractmethod

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession


class AbstractPipelineStep(ABC):
    def __call__(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        # cosa succede se si scatena una eccezione dentro process?
        self._preprocess(df)
        result = self._process(df, spark)
        return self._postprocess(result)

    @abstractmethod
    def _preprocess(self, df: DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, result: DataFrame) -> DataFrame:
        # chaching result, storing for debug, plotting
        raise NotImplementedError

    # @classmethod
    # @abstractmethod
    # def from_config(cls: Type['MyAbstractClass'], config: Any) -> Type['MyAbstractClass']:
    #    raise NotImplementedError
