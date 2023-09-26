from typing import Optional

from pyspark.sql import DataFrame
from pyspark.ml import Estimator
from pyspark.ml.feature import (
    StringIndexerModel,
    OneHotEncoderModel,
)
from pydantic import BaseModel

from ts_train.ft2model.training_helper import (
    Params,
    tune_parameters,
    get_estimator,
    get_evaluator,
    fit,
    predict,
    score,
    label_encode,
    one_hot_encode,
    vector_assemble,
    get_features_importance,
)


class ClassificationTraining(BaseModel):
    features_cols_name: list[str]
    label_col_name: str
    type: str
    params: Optional[Params] = None
    ordered_categ_features_cols_name: Optional[list[str]] = None
    unordered_categ_features_cols_name: Optional[list[str]] = None

    _estimator: Optional[Estimator] = None
    _best_params: Optional[Params] = None
    _string_indexer_model: Optional[StringIndexerModel] = None
    _one_hot_encoder_model: Optional[OneHotEncoderModel] = None

    def fit(self, df: DataFrame, metric="f1", cv_folds: int = 3) -> Estimator:
        df = self._preprocess_dataframe(df)

        if self._are_params_tunable():
            self._best_params, self._estimator = tune_parameters(
                df=df,
                params=self.params,  # type: ignore
                estimator=get_estimator(
                    type=self.type,
                    label_col_name=self.label_col_name,
                ),
                evaluator=get_evaluator(
                    metric=metric, label_col_name=self.label_col_name
                ),
                num_folds=cv_folds,
            )
        else:
            self._best_params = self.params
            self._estimator = fit(
                df=df,
                estimator=get_estimator(
                    label_col_name=self.label_col_name,
                    params=self._best_params,
                ),
            )

        return self._estimator

    def predict(self, df: DataFrame) -> DataFrame:
        self._check_model_already_fit()

        df = self._preprocess_dataframe(df)

        return predict(df=df, transformer=self._estimator)  # type: ignore

    def score(self, df: DataFrame, metric="f1") -> float:
        self._check_model_already_fit()

        if "prediction" not in df.columns or "rawPrediction" not in df.columns:
            df = self.predict(df=df)
        # TODO check that in df there is label_col_name

        return score(
            df=df,
            evaluator=get_evaluator(metric=metric, label_col_name=self.label_col_name),
        )

    def get_feature_importance(self) -> dict[str, float]:
        self._check_model_already_fit()

        return get_features_importance(
            model=self._estimator,  # type: ignore
            features_cols_name=self.features_cols_name,
        )

    def _are_params_tunable(self):
        if self.params is None:
            return False
        for param_value in self.params.values():
            if isinstance(param_value, list):
                return True
        return False

    def _check_model_already_fit(self):
        if self._estimator is None:
            raise ValueError("You should fit the model first.")

    def _preprocess_dataframe(self, df: DataFrame) -> DataFrame:
        if (
            self.ordered_categ_features_cols_name
            and len(self.ordered_categ_features_cols_name) > 0
        ):
            df, self._string_indexer_model = label_encode(
                df=df,
                cols_name=self.ordered_categ_features_cols_name,
                string_indexer_model=self._string_indexer_model,
            )

        if (
            self.unordered_categ_features_cols_name
            and len(self.unordered_categ_features_cols_name) > 0
        ):
            (
                df,
                self._one_hot_encoder_model,
                self._string_indexer_model,
            ) = one_hot_encode(
                df=df,
                cols_name=self.unordered_categ_features_cols_name,
                string_indexer_model=self._string_indexer_model,
                one_hot_encoder_model=self._one_hot_encoder_model,
            )

        df = vector_assemble(df=df, features_cols_name=self.features_cols_name)

        return df
