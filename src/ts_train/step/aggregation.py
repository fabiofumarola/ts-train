# from typing import Any, List, NewType, Union, Dict
# from pyspark.sql.dataframe import DataFrame
# from pyspark.sql import SparkSession
# from .core import AbstractPipelineStep


# class Aggregation(AbstractPipelineStep):
#     """
#     Create a new dataframe with new columns made by numeric cols using categorical
#     cols as filters

#     """

#     # Queste diventeranno dataclasses
#     Aggregation: NewType = Dict[str, Union[str, List[str]]]
#     FilterVariable: NewType = Dict[str, Union[str, List[str]]]
#     Filter: NewType = Dict[str, Union[str, FilterVariable]]

#     def __init__(
#         self,
#         time_bucket_col_name: str,
#         identifier_cols_name: Union[str, List[str]],
#         aggregations: List[Aggregation],
#         filters: List[Filter],
#     ) -> None:
#         pass

#     """
#     Che tipo di prerequisiti deve soddisfare per essere eseguito?
#     Cosa ritorna?
#     Cosa facciamo con quello che ritorna?

#     TEST DA EFFETTUARE

#     """

#     def _preprocess(self, df: DataFrame) -> None:
#         # CODICE DA ESEGUIRE ONLINE
#         # check if df is not empty
#         # check if there is a timebucket column name
#         # check if there are all the tidentifier cols names
#         # check if there is at least one aggregation, the dictionary must have at
#         # least
#         # variable and function (you can skip the filter)
#         # check that varaible in filter are real categoric columns
#         # check that numeric variable is a real numeric column
#         # check that function is a valid function (present in the enumeration)
#         # check that user provides all the information ot create filters
#         # check that indicated categorical values are actually present across the rows

#         # PYTEST:
#         # Fornire diversi input e vericare che tornino true o false
#         return df

#     def _process(self, df: DataFrame, spark: SparkSession) -> DataFrame:
#         # Creare un df nuovo con diverse colonne in base alle aggregazioni configurate

#         # PYTEST:
#         # possiamo calcolare a priori il numero di colonne che dovrebbe avere il df
#         # risultante?

#         # assert len(old_df) <= len(new_df)
#         # calcoalre a priori il numero di colonne che deve avere il df finale
#         # everificare che i conti tornino
#         # fare un caso in cui metti solo variabili nuemriche e verichi le colonne
#         # generate
#         # verificare la possibilità di fare aggregazione solo su variabili categoriche
#         return df

#     def _postprocess(self, result: DataFrame) -> DataFrame:
#         return result
