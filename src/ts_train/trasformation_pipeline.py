from typing import List

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession

from ts_train.step.core import AbstractPipelineStep


class TrasformationPipeline:
    def __init__(
        self,
        pipelineSteps: List[AbstractPipelineStep],
    ) -> None:
        pass

    def process(self, df: DataFrame, spark_session: SparkSession) -> DataFrame:
        # ciclo sulla lista di step, li esegue e passo ogni ottenuto allo step
        # successivo
        raise NotImplementedError
